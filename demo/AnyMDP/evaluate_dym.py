import os
import sys
import argparse
import torch
import numpy
import matplotlib.pyplot as plt
import csv
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict

package_path = '/home/shaopt/code/foundation_model'
sys.path.append(package_path)
from l3c_baselines.dataloader import AnyMDPDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import count_parameters, check_model_validity, model_path
from l3c_baselines.utils import Configure, gradient_failsafe, DistStatistics, rewards2go
from l3c_baselines.models import AnyMDPRSA

import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def create_env(env_name):
    if(env_name.lower() == "lake"):
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
        return env
    # elif(env_name.lower() == "cliff"):
    #     env = gym.make('CliffWalking-v0')
    #     return env
    # elif(env_name.lower() == "taxi"):
    #     env = gym.make('Taxi-v3')
    #     return env
    # elif(env_name.lower() == "blackjack"):
    #     env = gym.make('Blackjack-v1', natural=False, sab=True)
    #     return env
    else:
        raise ValueError("Unknown env name: {}".format(env_name))
def calculate_result_matrix(loss_matrix):
    """
    Calculate and return a new matrix, where the first row is the result of averaging the input matrix along dim 0,
    and the second row is the result of calculating the variance of the input matrix along dim 0.

    Parameters:
    loss_matrix (torch.Tensor): The input tensor, its shape should be [batch_size, seq_length].

    Returns:
    result_matrix (torch.Tensor): The output tensor, its shape is [2, seq_length].
    """
    # Calculate the mean and variance along dim 0
    mean_loss = []
    var_loss = []
    if loss_matrix.shape[0] > 1:
        mean_loss = torch.mean(loss_matrix, dim=0)
        var_loss = torch.var(loss_matrix, dim=0)
    else:
        mean_loss = loss_matrix
        var_loss = torch.zeros_like(mean_loss)

    # Create a new matrix
    result_matrix = torch.stack((mean_loss, var_loss), dim=0)

    return result_matrix

def string_mean_var(downsample_length, mean, var):
    string=""
    for i in range(mean.shape[0]):
        string += f'{downsample_length * i}\t{mean[i]}\t{var[i]}\n'
    return string

def anymdp_model_epoch(rank, env, task_num, config, model, main, device, downsample_length = 10):
    # Example training loop

    state_init, _ = env.reset()
    obs_arr = [state_init]
    act_arr = []
    rew_arr = []
    
    step = 1
    cache = None
    model.init_mem()

    for task_index in range(task_num):
        done = False
        while not done:
            
            
    # dataset = AnyMDPDataSet(config.data_path, config.seq_len, verbose=main)
    # dataloader = PrefetchDataLoader(dataset, batch_size=config.batch_size, rank=rank, world_size=world_size)
    # all_length = len(dataloader)

    # if config.downsample_size is not None:
    #     downsample_length = config.downsample_size

    # stat2 = DistStatistics("loss_wm_s_ds", "loss_wm_r_ds", "loss_pm_ds", "count", pointwise=True)
    # if(main):
    #     log_debug("Start evaluation ...")
    #     log_progress(0)
    # dataset.reset(0)

    # loss_wm_s_T_batch = None
    # loss_wm_r_T_batch = None
    # loss_pm_T_batch = None
    # loss_count_batch = 0
    # for batch_idx, batch in enumerate(dataloader):
    #     start_position = 0
    #     model.module.reset()
    #     sarr, baarr, laarr, rarr = batch
    #     r2goarr = rewards2go(rarr)
    #     loss_wm_s_T = None
    #     loss_wm_r_T = None
    #     loss_pm_T = None      
    #     for sub_idx, states, bactions, lactions, rewards, r2go in segment_iterator(
    #                 config.seq_len, config.seg_len, device, 
    #                 (sarr, 1), baarr, laarr, rarr, (r2goarr, 1)):
    #         with torch.no_grad():
    #             # loss dim is Bxt, t = T // seg_len 
    #             loss = model.module.sequential_loss(
    #                         r2go[:, :-1],
    #                         states, 
    #                         rewards, 
    #                         bactions, 
    #                         lactions, 
    #                         r2go[:, 1:], 
    #                         start_position=start_position,
    #                         reduce_dim=None)
    #         if (sub_idx == 0):
    #             loss_wm_s_T=torch.nan_to_num(loss["wm-s"], nan=0.0)
    #             loss_wm_r_T=torch.nan_to_num(loss["wm-r"], nan=0.0)
    #             loss_pm_T=torch.nan_to_num(loss["pm"], nan=0.0)
    #         else:
    #             loss_wm_s_T = torch.cat((loss_wm_s_T, torch.nan_to_num(loss["wm-s"], nan=0.0)), dim=1)
    #             loss_wm_r_T = torch.cat((loss_wm_r_T, torch.nan_to_num(loss["wm-r"], nan=0.0)), dim=1)
    #             loss_pm_T = torch.cat((loss_pm_T, torch.nan_to_num(loss["pm"], nan=0.0)), dim=1)
    #         start_position += bactions.shape[1]
        
    #     # Append over all segment, loss_wm_s_arr, loss_wm_r_arr and loss_pm_arr dim become BxT
    #     # Downsample over T, dim become [B,T//downsample_length]
    #     dim_1, seq_length = loss_wm_s_T.shape
    #     num_elements_to_keep = seq_length // downsample_length * downsample_length
    #     loss_wm_s_ds = torch.mean(loss_wm_s_T[:, :num_elements_to_keep].view(dim_1, seq_length//downsample_length, -1), dim=2)
    #     loss_wm_r_ds = torch.mean(loss_wm_r_T[:, :num_elements_to_keep].view(dim_1, seq_length//downsample_length, -1), dim=2)
    #     loss_pm_ds = torch.mean(loss_pm_T[:, :num_elements_to_keep].view(dim_1, seq_length//downsample_length, -1), dim=2)
        
    #     loss_wm_s_T_batch = torch.cat((loss_wm_s_T_batch, loss_wm_s_ds), dim=0) if loss_wm_s_T_batch is not None else loss_wm_s_ds
    #     loss_wm_r_T_batch = torch.cat((loss_wm_r_T_batch, loss_wm_r_ds), dim=0) if loss_wm_r_T_batch is not None else loss_wm_r_ds
    #     loss_pm_T_batch = torch.cat((loss_pm_T_batch, loss_pm_ds), dim=0) if loss_pm_T_batch is not None else loss_pm_ds
    #     loss_count_batch += dim_1

        
        

    #     if(main):
    #         log_progress((batch_idx + 1) / all_length)

    # # finish batch loop
    # # Calculate result matrix, dim become [2,T//downsample_length], first row is position_wise mean, second row is variance.
    # # Get the result in each device
    # stat_loss_wm_s = calculate_result_matrix(loss_wm_s_T_batch)
    # stat_loss_wm_r = calculate_result_matrix(loss_wm_r_T_batch)
    # stat_loss_pm = calculate_result_matrix(loss_pm_T_batch)
    # # Merge the result accross all device
    # stat2.append_with_safety(rank, 
    #                         loss_wm_s_ds=stat_loss_wm_s, 
    #                         loss_wm_r_ds=stat_loss_wm_r, 
    #                         loss_pm_ds=stat_loss_pm,
    #                         count=torch.tensor(loss_count_batch))
    # if(main):
    #     print("------------Debug: stat_loss_wm_s =",stat_loss_wm_s)
    #     print("------------Debug: stat_loss_wm_r =",stat_loss_wm_r)
    #     print("------------Debug: stat_loss_pm =",stat_loss_pm)
    #     print("------------Debug: loss_count_batch =",loss_count_batch)
    

    # if(main):
        # if not os.path.exists(config.output):
        #     os.makedirs(config.output)
        # stat_res_wm_s = string_mean_var(downsample_length, stat2()["loss_wm_s_ds"][0], stat2()["loss_wm_s_ds"][1])
        # file_path = f'{config.output}/position_wise_wm_s.txt'
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # with open(file_path, 'w') as f_model:
        #     f_model.write(stat_res_wm_s)
        
        # stat_res_wm_r = string_mean_var(downsample_length, stat2()["loss_wm_r_ds"][0], stat2()["loss_wm_r_ds"][1])
        # file_path = f'{config.output}/position_wise_wm_r.txt'
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # with open(file_path, 'w') as f_model:
        #     f_model.write(stat_res_wm_r)
        
        # stat_res_pm = string_mean_var(downsample_length, stat2()["loss_pm_ds"][0], stat2()["loss_pm_ds"][1])
        # file_path = f'{config.output}/position_wise_pm.txt'
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        # with open(file_path, 'w') as f_model:
        #     f_model.write(stat_res_pm)
    #stat2.reset()

def anymdp_main_epoch(rank, use_gpu, world_size, config, main_rank, run_name):
    
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if(main_rank is None):
        main = False
    elif(main_rank == "all" or main_rank == rank):
        main = True
    else:
        main = False
    if(main):
        print("Main gpu", use_gpu, "rank:", rank, device)

    test_config = config.test_config
    train_config = config.train_config
    
    # Load Model
    model = AnyMDPRSA(config.model_config, verbose=main)
    model = model.to(device)
    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    if(test_config.has_attr("load_model_path") and 
            test_config.load_model_path is not None and 
            test_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, f'{test_config.load_model_path}/model.pth', 
                                  black_list=train_config.load_model_parameter_blacklist, 
                                  strict_check=False)
        print("------------Load model success!------------")

    # Perform the first evaluation
    anymdp_model_epoch(rank, world_size, test_config, model, main, device, test_config.downsample_size)

    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, help="YAML configuration file")
    parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    config = Configure()
    config.from_yaml(args.configuration)

    # Get the dictionary of attributes
    if args.configs:
        for pair in args.configs:
            key, value = pair.split('=')
            config.set_value(key, value)
            print(f"Rewriting configurations from args: {key} to {value}")
    print("Final configuration:\n", config)
    demo_config = config.demo_config
    os.environ['MASTER_PORT'] = demo_config.master_port        # Example port, choose an available port

    # mp.spawn(anymdp_main_epoch,
    #          args=(use_gpu, world_size, config, 0, config.run_name),
    #          nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
    #          join=True)
    env = create_env(demo_config.env_config.name)
    env.reset()
    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)
    MAX_ITERATIONS = 10
    for i in range(MAX_ITERATIONS):
        random_action = env.action_space.sample()
        print("Random action: ", random_action)
        new_state, reward, done, info = env.step(random_action)
        print("New state: ", new_state)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Info: ", info)
        env.render()
        if done:
            break