import argparse
import pickle
import numpy
import random
import os
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from l3c.anymdpv2 import AnyMDPv2TaskSampler
from l3c.anymdpv2 import AnyMDPEnv

class RLCoach:
    def __init__(self, env, n_epochs, seed=None):
        self.env = env
        self.seed = seed
        self.n_epochs = n_epochs
        
        # 初始化基础策略
        self.policies = {
            "random": lambda x: env.action_space.sample(),
            "ppo_mlp": PPO(
                "MlpPolicy", 
                env, 
                seed=seed,
                verbose=0,
                learning_rate=3e-4,
                batch_size=64,
                gamma=0.99,
            ),
            "ppo_rnn": RecurrentPPO(
                "MlpLstmPolicy",
                env,
                verbose=0,
                seed=seed,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                policy_kwargs=dict(
                    lstm_hidden_size=32,
                    n_lstm_layers=2,
                    enable_critic_lstm=True,
                    net_arch=dict(
                        pi=[64, 32],
                        vf=[64, 32]
                    ),
                    activation_fn=nn.ReLU
                ),
            )
        }
        
        # LSTM状态
        self.lstm_states = None
        self.episode_starts = True
        
        # 数据存储
        self.episode_data = []  # [(states, actions, rewards, policy_type, success), ...]
        self.task_configs = []
        
    def reset(self):
        """重置LSTM状态"""
        self.lstm_states = None
        self.episode_starts = True
        
    def train_ppo_policies(self, max_steps_per_epoch, max_episodes_per_epoch, train_epochs=10):
        """训练PPO策略"""
        print("Training PPO policies...")
        
        # 只训练PPO策略
        ppo_policies = {name: policy for name, policy in self.policies.items() 
                       if isinstance(policy, (PPO, RecurrentPPO))}
        
        for policy_name, policy in ppo_policies.items():
            print(f"\nTraining {policy_name}...")
            
            for epoch in range(train_epochs):
                episode_count = 0
                total_steps = 0
                total_success = 0
                episode_returns = []
                
                while episode_count < max_episodes_per_epoch and total_steps < max_steps_per_epoch:
                    state, info = self.env.reset()
                    self.reset()
                    
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        # 训练时使用随机动作
                        if isinstance(policy, RecurrentPPO):
                            action, self.lstm_states = policy.predict(
                                state,
                                state=self.lstm_states,
                                episode_start=self.episode_starts,
                                deterministic=False
                            )
                            self.episode_starts = False
                        else:
                            action, _ = policy.predict(state, deterministic=False)
                        
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        
                        # 存入策略的replay buffer
                        policy.replay_buffer.add(
                            obs=state,
                            action=action,
                            reward=reward,
                            next_obs=next_state,
                            done=done
                        )
                        
                        episode_reward += reward
                        total_steps += 1
                        state = next_state
                    
                    success = terminated and reward > 0
                    total_success += int(success)
                    episode_returns.append(episode_reward)
                    episode_count += 1
                
                # 训练策略
                policy.train()
                
                # 打印训练信息
                print(f"Training Epoch {epoch}: episodes={episode_count}, "
                    f"steps={total_steps}, "
                    f"success_rate={total_success/episode_count:.2%}, "
                    f"avg_return={numpy.mean(episode_returns):.2f}")
    
    def collect_data(self, max_steps_per_epoch, max_episodes_per_epoch):
        """使用训练好的策略收集数据"""
        print("\nCollecting data using trained policies...")
        
        for policy_name, policy in self.policies.items():
            print(f"\nUsing policy: {policy_name}")
            
            for epoch in range(self.n_epochs):
                episode_count = 0
                total_steps = 0
                total_success = 0
                episode_returns = []
                
                while episode_count < max_episodes_per_epoch and total_steps < max_steps_per_epoch:
                    # 重置环境和状态
                    state, info = self.env.reset()
                    current_task_config = {
                        "mode": self.env.task.mode,
                        "ndim": self.env.task.ndim,
                        "state": self.env.task.state
                    }
                    self.reset()
                    
                    # 记录单个episode的数据
                    episode_states = []
                    episode_actions = []
                    episode_rewards = []
                    
                    done = False
                    while not done:
                        # 使用确定性策略收集数据
                        if isinstance(policy, type(lambda:0)):  # random policy
                            action = policy(state)
                        elif isinstance(policy, RecurrentPPO):
                            action, self.lstm_states = policy.predict(
                                state,
                                state=self.lstm_states,
                                episode_start=self.episode_starts,
                                deterministic=True
                            )
                            self.episode_starts = False
                        else:
                            action, _ = policy.predict(state, deterministic=True)
                        
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        
                        episode_states.append(state)
                        episode_actions.append(action)
                        episode_rewards.append(reward)
                        
                        total_steps += 1
                        state = next_state
                    
                    success = terminated and reward > 0
                    total_success += int(success)
                    
                    # 保存episode数据
                    self.episode_data.append((
                        numpy.array(episode_states),
                        numpy.array(episode_actions),
                        numpy.array(episode_rewards),
                        policy_name,
                        success
                    ))
                    self.task_configs.append(current_task_config)
                    episode_returns.append(sum(episode_rewards))
                    
                    episode_count += 1
                
                # 打印数据收集信息
                print(f"Collection Epoch {epoch}: episodes={episode_count}, "
                    f"steps={total_steps}, "
                    f"success_rate={total_success/episode_count:.2%}, "
                    f"avg_return={numpy.mean(episode_returns):.2f}")
    
    def save(self, path):
        """保存收集的数据和任务配置"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 计算每个episode的得分
        episode_scores = []
        for i, (states, actions, rewards, policy_type, success) in enumerate(self.episode_data):
            score = sum(rewards) + (100 if success else 0)
            episode_scores.append((i, score))
        
        # 按得分排序并获取索引
        sorted_indices = [i for i, _ in sorted(episode_scores, 
                                            key=lambda x: x[1], 
                                            reverse=True)]
        
        # 根据比例划分数据(前20%最好,中间50%中等,后30%最差)
        n_total = len(sorted_indices)
        n_good = int(0.2 * n_total)
        n_medium = int(0.5 * n_total)
        
        good_indices = sorted_indices[:n_good]
        medium_indices = sorted_indices[n_good:n_good + n_medium]
        poor_indices = sorted_indices[n_good + n_medium:]
        
        # 按组织重排数据
        grouped_data = {
            "good": {
                "episode_data": [self.episode_data[i] for i in good_indices],
                "task_configs": [self.task_configs[i] for i in good_indices],
                "scores": [episode_scores[i][1] for i in good_indices]
            },
            "medium": {
                "episode_data": [self.episode_data[i] for i in medium_indices],
                "task_configs": [self.task_configs[i] for i in medium_indices],
                "scores": [episode_scores[i][1] for i in medium_indices]
            },
            "poor": {
                "episode_data": [self.episode_data[i] for i in poor_indices],
                "task_configs": [self.task_configs[i] for i in poor_indices],
                "scores": [episode_scores[i][1] for i in poor_indices]
            }
        }
        
        # 保存分组数据
        save_data = {
            "grouped_data": grouped_data,
            "stats": {
                "total_episodes": n_total,
                "group_sizes": {
                    "good": len(good_indices),
                    "medium": len(medium_indices),
                    "poor": len(poor_indices)
                },
                "score_ranges": {
                    "good": (min(grouped_data["good"]["scores"]), 
                            max(grouped_data["good"]["scores"])),
                    "medium": (min(grouped_data["medium"]["scores"]), 
                            max(grouped_data["medium"]["scores"])),
                    "poor": (min(grouped_data["poor"]["scores"]), 
                            max(grouped_data["poor"]["scores"]))
                }
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
            
        # 保存策略模型
        for name, policy in self.policies.items():
            if isinstance(policy, (PPO, RecurrentPPO)):
                policy.save(f"{path[:-4]}_{name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                       help="path to save the trained coach")
    parser.add_argument("--state_dim", type=int, default=256,
                       help="state dimension")
    parser.add_argument("--action_dim", type=int, default=256,
                       help="action dimension")
    parser.add_argument("--ndim", type=int, default=8,  
                       help="ndim for task sampler")
    parser.add_argument("--mode", type=str, default=None, 
                   choices=["static", "dynamic", "universal"],
                   help="task mode (if not specified, will be randomly sampled)")
    parser.add_argument("--max_steps", type=int, default=4000,
                       help="maximum steps per epoch")
    parser.add_argument("--max_episodes", type=int, default=20,
                       help="maximum episodes per epoch")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="number of epochs")
    parser.add_argument("--train_epochs", type=int, default=10,
                       help="number of epochs for training PPO policies")
    parser.add_argument("--seed", type=int, default=None,
                       help="random seed")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
    
    # 创建环境
    env = AnyMDPEnv(max_steps=args.max_steps)
    task = AnyMDPv2TaskSampler(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        ndim=args.ndim,  
        mode=args.mode,
        seed=args.seed
    )
    env.set_task(task)
    
    # 创建并训练coach
    coach = RLCoach(env, args.n_epochs, seed=args.seed)
    
    # 先训练PPO策略
    coach.train_ppo_policies(args.max_steps, args.max_episodes, args.train_epochs)
    
    # 收集数据
    coach.collect_data(args.max_steps, args.max_episodes)
    
    # 保存数据
    coach.save(args.save_path)