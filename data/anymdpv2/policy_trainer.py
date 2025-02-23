import argparse
import pickle
import numpy
import random
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from l3c.anymdpv2 import AnyMDPv2TaskSampler, AnyMDPEnv

class PolicyTrainer:
    def __init__(self, data_path, seed=None):
        # 加载分组数据
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.grouped_data = data["grouped_data"]
        
        # 从数据中获取mode
        sample_task_config = data["grouped_data"]["good"]["task_configs"][0]
        self.mode = sample_task_config["mode"]  # 记录mode
        
        # 设置随机种子
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
        
        # 从数据中获取环境参数
        state_dim = sample_task_config["state"].shape[0]
        
        # 创建环境
        self.env = AnyMDPEnv(max_steps=4000)
        self.task = AnyMDPv2TaskSampler(
            state_dim=state_dim,
            action_dim=state_dim,
            ndim=sample_task_config["ndim"],
            mode=self.mode,  # 使用数据中的mode
            seed=seed
        )
        self.env.set_task(self.task)
        
        # 为每组数据创建coach
        self.coaches = {
            "good": self.create_coach(),
            "medium": self.create_coach(),
            "poor": self.create_coach()
        }
    
    def create_coach(self):
        """创建一个新的coach"""
        return PPO(
            "MlpPolicy", 
            self.env, 
            seed=self.seed,
            verbose=0,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[64, 32],
                    vf=[64, 32]
                ),
                activation_fn=nn.ReLU
            )
        )
    
    def train(self, max_steps=4000, max_episodes=20, train_epochs=10):
        """训练所有coach"""
        for group_name, coach in self.coaches.items():
            print(f"\nTraining {group_name} coach...")
            data = self.grouped_data[group_name]
            
            for epoch in range(train_epochs):
                episode_count = 0
                total_steps = 0
                total_success = 0
                episode_returns = []
                
                # 用分组数据训练coach
                for episode_data, task_config in zip(data["episode_data"], data["task_configs"]):
                    if episode_count >= max_episodes or total_steps >= max_steps:
                        break
                        
                    # 验证mode一致性
                    if task_config["mode"] != self.mode:
                        raise ValueError(f"Task mode {task_config['mode']} does not match data mode {self.mode}")
                    
                    states, actions, rewards, _, success = episode_data
                    
                    # 设置任务配置
                    self.task.set_config(task_config)
                    self.env.set_task(self.task)
                    
                    # 训练coach
                    episode_reward = 0
                    for state, action, reward in zip(states, actions, rewards):
                        coach.replay_buffer.add(
                            obs=state,
                            action=action,
                            reward=reward,
                            next_obs=state,  # 简化处理
                            done=False
                        )
                        episode_reward += reward
                        total_steps += 1
                    
                    # 记录统计信息
                    total_success += int(success)
                    episode_returns.append(episode_reward)
                    episode_count += 1
                    
                    # 训练
                    coach.train()
                
                # 打印训练信息
                print(f"Training Epoch {epoch}: episodes={episode_count}, "
                      f"steps={total_steps}, "
                      f"success_rate={total_success/episode_count:.2%}, "
                      f"avg_return={numpy.mean(episode_returns):.2f}")
    
    def save(self, save_path):
        """保存训练好的coaches"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        behavior_dict = {
            name: coach.policy.state_dict()
            for name, coach in self.coaches.items()
        }
        
        reference_dict = {
            "policy": self.coaches["good"].policy.state_dict(),
            "info": {
                "type": "best_coach",
                "performance": "good",
                "mode": self.mode,  # 记录mode信息
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "batch_size": 64,
                    "gamma": 0.99,
                    "net_arch": {
                        "pi": [64, 32],
                        "vf": [64, 32]
                    }
                }
            }
        }
        
        torch.save({
            "behavior_dict": behavior_dict,
            "reference_dict": reference_dict,
            "mode": self.mode  
        }, save_path)
    
        # 同时保存完整的coach模型
        for name, coach in self.coaches.items():
            coach.save(f"{save_path[:-4]}_{name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                       help="path to raw data file")
    parser.add_argument("--save_path", type=str, required=True,
                       help="path to save trained coaches")
    parser.add_argument("--max_steps", type=int, default=4000,
                       help="maximum steps per epoch")
    parser.add_argument("--max_episodes", type=int, default=20,
                       help="maximum episodes per epoch")
    parser.add_argument("--train_epochs", type=int, default=10,
                       help="number of training epochs")
    parser.add_argument("--seed", type=int, default=None,
                       help="random seed")
    args = parser.parse_args()
    
    trainer = PolicyTrainer(args.data_path, seed=args.seed)
    trainer.train(args.max_steps, args.max_episodes, args.train_epochs)
    trainer.save(args.save_path)