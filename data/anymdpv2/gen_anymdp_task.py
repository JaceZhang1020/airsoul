import argparse
import pickle
import random
from l3c.utils import pseudo_random_seed
from l3c.anymdpv2 import AnyMDPv2TaskSampler


if __name__=="__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", type=int, default=256, help="dimension of observation space")
    parser.add_argument("--action_dim", type=int, default=256, help="dimension of action space") 
    parser.add_argument("--ndim", type=int, default=None, help="dimension of inner state space (default: random 4-16)")
    parser.add_argument("--mode", type=str, choices=["static", "dynamic", "universal"], default=None, 
                        help="task mode (default: random choice)")
    parser.add_argument("--task_number", type=int, default=1, help="number of tasks to generate")
    parser.add_argument("--output_path", type=str, required=True, help="output file path")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        pseudo_random_seed(args.seed)

    # Generate tasks
    tasks = []
    for idx in range(args.task_number):
        task = AnyMDPv2TaskSampler(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            ndim=args.ndim,      # 将会在TaskSampler中处理None的情况
            mode=args.mode,      # 将会在TaskSampler中处理None的情况
            seed=None if args.seed is None else args.seed + idx
        )
        tasks.append(task)
        print(f"Generated task {idx+1}/{args.task_number}")

    # Save tasks
    output_file = args.output_path if args.output_path.endswith('.pkl') else args.output_path + '.pkl'
    print(f"Writing {len(tasks)} tasks to {output_file}")
    print(f"Configuration:")
    print(f"  state_dim={args.state_dim}, action_dim={args.action_dim}")
    print(f"  ndim={'random 4-16' if args.ndim is None else args.ndim}")
    print(f"  mode={'random choice' if args.mode is None else args.mode}")
    
    with open(output_file, 'wb') as fw:
        pickle.dump(tasks, fw)