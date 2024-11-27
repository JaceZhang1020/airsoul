import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from l3c_baselines.dataloader import segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import Configure, DistStatistics, rewards2go, downsample
from l3c_baselines.utils import EpochManager, GeneratorBase, Logger
from l3c_baselines.dataloader import AnyMDPDataSet

import gym
import numpy
import pickle
from gym.envs.toy_text.frozen_lake import generate_random_map
from l3c.anymdp import AnyMDPTaskSampler


def string_mean_var(downsample_length, res):
    string=""
    if(numpy.size(res["mean"]) > 1):
        for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
            string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    else:
        string =  f'{0}\t{res["mean"]}\t{res["bound"]}\n'
    return string

@EpochManager
class AnyMDPEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=AnyMDPDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics()
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_reward_pred", 
                        "validation_policy"]
            self.stat = DistStatistics()
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
        if(self.config.has_attr('state_dropout')):
            self.state_dropout = self.config.state_dropout
        else:
            self.state_dropout = 0.20
        if(self.config.has_attr('reward_dropout')):
            self.reward_dropout = self.config.reward_dropout
        else:
            self.reward_dropout = 0.20

    def compute(self, sarr, baarr, laarr, rarr, 
                        epoch_id=-1, 
                        batch_id=-1):
        """
        Defining the computation function for each batch
        """
        state_dropout = 0.0
        reward_dropout = 0.0
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"
            state_dropout = self.state_dropout
            reward_dropout = self.reward_dropout
        else:
            state_dropout = 0.0
            reward_dropout = 0.0

        losses = []
        r2goarr = rewards2go(rarr)
        for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device, 
                    (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
            loss = self.model.module.sequential_loss(
                    r2go[:, :-1], # Prompts
                    states, 
                    rewards, # Rewards 
                    bactions, 
                    lactions, 
                    state_dropout=state_dropout, 
                    reward_dropout=reward_dropout,
                    use_loss_weight=self.is_training,
                    reduce_dim=self.reduce) # Do not use loss weight for evaluation
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm-s"]
                        + self.config.lossweight_worldmodel_rewards * loss["wm-r"]
                        + self.config.lossweight_entropy * loss["ent"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    loss_worldmodel_state = loss["wm-s"] / loss["count_s"],
                    loss_worldmodel_reward = loss["wm-r"] / loss["count_s"],
                    loss_policymodel = loss["pm"] / loss["count_a"],
                    entropy = -loss["ent"] / loss["count_a"],
                    count = loss["count_a"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["loss_worldmodel_state"]["mean"], 
                        stat_res["loss_worldmodel_reward"]["mean"], 
                        stat_res["loss_policymodel"]["mean"], 
                        stat_res["entropy"]["mean"],
                        epoch=epoch_id,
                        iteration=batch_id)
        else:
            loss_wm_s = torch.cat([loss["wm-s"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["wm-r"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            counts = torch.cat([loss["count_a"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]

            loss_wm_s = downsample(loss_wm_s, self.downsample_length)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            counts = downsample(counts, self.downsample_length)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        count=counts[i])
            
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_reward_pred"]["mean"], 
                        stat_res["validation_policy"]["mean"],
                        epoch=epoch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    for key_name in stat_res:
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)


class AnyMDPGenerator(GeneratorBase):
    def preprocess(self):
        if(self.config.env.lower().find("lake") >= 0):
            self.task_sampler = self.task_sampler_lake
        elif(self.config.env.lower().find("anymdp") >= 0):
            self.env = gym.make("anymdp-v0", max_steps=self.max_steps)
            self.task_sampler = self.task_sampler_anymdp
        else:
            log_fatal("Unsupported environment:", self.config.env)

        if(self.config.has_attr("task_file")):
            with open(self.config.task_file, 'rb') as fr:
                self.tasks = pickle.load(fr)
            log_debug(f"Read tasks from {self.config.task_file} success")
        else:
            self.tasks = None

        logger_keys = ["reward", "state_prediction", "reward_prediction", "success_rate"]

        self.stat = DistStatistics()
        self.logger = Logger("steps",
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)

    def task_sampler_anymdp(self, epoch_id=0):
        task_id = None
        if(self.tasks is None):
            dims = self.config.env.lower().replace("anymdp", "").split("x")
            task = AnyMDPTaskSampler(int(dims[0]), int(dims[1]))
        else:
            task_num = len(self.tasks)
            task_id = (epoch_id * self.world_size + self.rank) % task_num
            task = self.tasks[task_id]
        self.env.set_task(task)
        return task_id

    def task_sampler_lake(self, epoch_id=0):
        self.env = gym.make(
            'FrozenLake-v1', 
            map_name=self.config.env.replace("lake", ""), 
            is_slippery=True, 
            max_episode_steps=1000)
        return None

    def is_success_fail(self, reward, done):
        if(reward > 1.0e-3 and done):
            return 1
        elif(done):
            return -1
        else:
            return 0
        
    def in_context_learn_from_teacher(self, task_id=None):
        # Task ID: retrieve the correpsonding teacher trajectory with task ID
        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = numpy.load(os.path.join(folder_path, 'observations.npy'))
                actions = numpy.load(os.path.join(folder_path, 'actions_behavior.npy'))
                rewards = numpy.load(os.path.join(folder_path, 'rewards.npy'))
                states = states.astype(numpy.int32)
                actions = actions.astype(numpy.int32)
                rewards = rewards.astype(numpy.float32)
                segment_len = 1000
                for start in range(0, len(states), segment_len):
                    end = min(start + segment_len, len(states))
                    self.model.module.in_context_learn(
                        None,
                        states[start:end],
                        actions[start:end],
                        rewards[start:end],
                        single_batch=True,
                        single_step=False)
        print("Finish Learning.")

    def __call__(self, epoch_id):
        obs_arr = []
        act_arr = []
        rew_arr = []
        
        reward_error = []
        state_error = []

        step = 0
        trail = 0

        task_id = self.task_sampler(epoch_id=epoch_id)

        self.model.eval()

        if self.config.learn_from_data:
            self.in_context_learn_from_teacher(task_id=task_id)

        pred_state_dist = None
        is_succ = 0
        is_fail = 0
        success_rate = []

        while trail < self.max_trails and step < self.max_steps:
            done = False
            previous_state, _ = self.env.reset()
            obs_arr.append(previous_state)
            if(pred_state_dist is not None):
                state_error.append(-numpy.log(pred_state_dist[int(previous_state)].item()))
            
            epoch_start_step = step
            while not done:
                pred_state_dist, action, pred_reward = self.model.module.generate(
                    None,
                    previous_state,
                    action_clip=self.config.action_clip,
                    temp=self._scheduler(step))
                                
                # interact with env
                new_state, new_reward, done, *_ = self.env.step(action)

                # collect data
                act_arr.append(action)
                rew_arr.append(new_reward)
                # world model reward prediction correct count:
                # reward_correct_prob += reward_out_prob_list[0,0, int(new_reward)].item()

                # start learning
                self.model.module.in_context_learn(
                    None,
                    previous_state,
                    action,
                    new_reward)

                obs_arr.append(new_state) 
                state_error.append(-numpy.log(pred_state_dist[int(new_state)].item()))
                if(done):
                    act_arr.append(self.env.action_space.n)
                    rew_arr.append(0)

                succ_fail = self.is_success_fail(new_reward, done)
                is_succ += (succ_fail > 0)
                is_fail += (succ_fail < 0)
                success_rate.append(is_succ / (is_succ + is_fail + 1.0e-6))

                reward_error.append((new_reward - pred_reward) ** 2)

                # Judge if success
                
                previous_state = new_state
                
                step += 1
                if(step > self.max_steps):
                    break
            trail += 1
            self.logger(step,
                        numpy.mean(rew_arr[epoch_start_step:]), 
                        numpy.mean(state_error[epoch_start_step:]), 
                        numpy.mean(reward_error[epoch_start_step:]),
                        success_rate[-1])

        ds_state_err = downsample(state_error, self.config.downsample_length)
        ds_reward_err = downsample(reward_error, self.config.downsample_length)
        ds_rewards = downsample(rew_arr, self.config.downsample_length)
        ds_success = downsample(success_rate, self.config.downsample_length)

        self.stat.gather(self.device,
                         reward=ds_rewards,
                         state_prediction=ds_state_err,
                         reward_prediction=ds_reward_err,
                         success_rate = ds_success)
    
    def postprocess(self):
        results=self.stat()
        self.logger("Final_Result",
                    results['reward']['mean'], 
                    results['state_prediction']['mean'], 
                    results['reward_prediction']['mean'],
                    results['success_rate']['mean'])
        if(self.config.has_attr("output")):
            if not os.path.exists(self.config.output):
                os.makedirs(self.config.output)
            for key_name in results:
                res_text = string_mean_var(self.config.downsample_length, results[key_name])

                file_path = f'{self.config.output}/result_{key_name}.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                with open(file_path, 'w') as f_model:
                    f_model.write(res_text)
