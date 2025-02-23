import argparse
import numpy
import random
import os
import torch
import multiprocessing
from l3c.anymdpv2 import AnyMDPv2TaskSampler, AnyMDPEnv
from tag_vocab import tag_mapping_id

def create_directory(path):
    os.makedirs(path, exist_ok=True)

class DataGenerator:
    def __init__(self, coach_path, seed=None):
        # 加载coach数据
        data = torch.load(coach_path)
        self.behavior_dict = data["behavior_dict"]
        self.reference_dict = data["reference_dict"]
        self.mode = data.get("mode")
        self.mask_all_tag_prob = 0.15
        self.mask_epoch_tag_prob = 0.15
        
        # 设置随机种子
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        
        # 创建环境
        self.env = AnyMDPEnv()
        self.task = AnyMDPv2TaskSampler(mode=self.mode)
        self.env.set_task(self.task)
        
        # 加载coaches
        self.coaches = {}
        for name in ["good", "medium", "poor"]:
            self.coaches[name] = torch.load(f"{coach_path[:-4]}_{name}")
    
    def generate_data(self, epoch_id, max_steps):
        """生成单个epoch的数据"""
        all_data = {
            "states": [],
            "actions_behavior": [],
            "actions_label": [],
            "rewards": [],
            "prompts": [],
            "tags": []
        }
        
        mask_all_tag = (random.random() < self.mask_all_tag_prob)
        mask_epoch_tag = (random.random() < self.mask_epoch_tag_prob)
        
        steps = 0
        total_reward = 0
        while steps < max_steps:
            state, _ = self.env.reset()
            
            behavior_name = random.choice(["good", "medium", "poor"])
            behavior_coach = self.coaches[behavior_name]
            
            done = False
            while not done and steps < max_steps:
                behavior_action = behavior_coach.predict(state, deterministic=True)[0]
                reference_action = self.coaches["good"].predict(state, deterministic=True)[0]
                
                next_state, reward, terminated, truncated, info = self.env.step(behavior_action)
                done = terminated or truncated
                
                # 确定tag
                if mask_all_tag or mask_epoch_tag:
                    tag = tag_mapping_id['unk']  # 3
                else:
                    tag = tag_mapping_id[behavior_name]  # poor->0, medium->1, good->2
                
                # prompt与tag使用相同的映射
                prompt = tag_mapping_id[behavior_name]  # poor->0, medium->1, good->2
                
                all_data["states"].append(state)
                all_data["actions_behavior"].append(behavior_action)
                all_data["actions_label"].append(reference_action)
                all_data["rewards"].append(reward)
                all_data["prompts"].append(prompt)
                all_data["tags"].append(tag)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:  
                    all_data["states"].append(next_state)
                    all_data["actions_behavior"].append(0)
                    all_data["actions_label"].append(0)
                    all_data["rewards"].append(0.0)
                    all_data["prompts"].append(tag_mapping_id['unk'])  # 3
                    all_data["tags"].append(tag_mapping_id['unk'])    # 3
                    
            mask_epoch_tag = (random.random() < self.mask_epoch_tag_prob)
        
        print(f"Finish running {epoch_id:06d}, sum reward: {total_reward:.6f}, steps: {steps}")
        
        return {k: numpy.array(v) for k, v in all_data.items()}

def dump_anymdp(work_id, workers, path_name, coach_path, max_steps, epoch_range, seed=None):
    generator = DataGenerator(coach_path, seed=seed)
    
    for epoch_id in epoch_range:
        results = generator.generate_data(epoch_id, max_steps)
        
        file_path = f'{path_name}/record-{epoch_id:06d}'
        create_directory(file_path)
        
        numpy.save(f"{file_path}/observations.npy", results["states"])
        numpy.save(f"{file_path}/actions_behavior.npy", results["actions_behavior"])
        numpy.save(f"{file_path}/actions_label.npy", results["actions_label"])
        numpy.save(f"{file_path}/rewards.npy", results["rewards"])
        numpy.save(f"{file_path}/prompts.npy", results["prompts"])
        numpy.save(f"{file_path}/tags.npy", results["tags"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./anymdp_data/",
                       help="output directory")
    parser.add_argument("--coach_path", type=str, required=True,
                       help="path to the trained coach")
    parser.add_argument("--max_steps", type=int, default=4000,
                       help="maximum steps per epoch")
    parser.add_argument("--epochs", type=int, default=1,
                       help="number of epochs")
    parser.add_argument("--start_index", type=int, default=0,
                       help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4,
                       help="number of multiprocessing workers")
    parser.add_argument("--seed", type=int, default=None,
                       help="random seed")
    args = parser.parse_args()

    # 多进程生成数据
    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)
        
        print(f"start processes generating {n_b:04d} to {n_e:04d}")
        process = multiprocessing.Process(
            target=dump_anymdp,
            args=(
                worker_id,
                args.workers,
                args.output_path,
                args.coach_path,
                args.max_steps,
                range(n_b, n_e),
                args.seed
            )
        )
        processes.append(process)
        process.start()
        
        n_b_t = n_e_t
    
    for process in processes:
        process.join()