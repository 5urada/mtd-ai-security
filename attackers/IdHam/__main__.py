import argparse
import sys

def main():
    """Main entry point for the ID-HAM simulator."""
    parser = argparse.ArgumentParser(description="ID-HAM MDP Simulator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ID-HAM policy")
    train_parser.add_argument("--runs_dir", type=str, default="runs/exp1")
    train_parser.add_argument("--seed", type=int, default=1337)
    train_parser.add_argument("--total_steps", type=int, default=100000)
    train_parser.add_argument("--attacker", type=str, default="static", 
                             choices=["static", "sequential", "dynamic"])
    train_parser.add_argument("--N", type=int, default=24, help="Number of hosts")
    train_parser.add_argument("--B", type=int, default=6, help="Number of blocks")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate sensitivity")
    eval_parser.add_argument("--ckpt", type=str, required=True)
    eval_parser.add_argument("--samples", type=int, default=200)
    eval_parser.add_argument("--attacker", type=str, default="static",
                            choices=["static", "sequential", "dynamic"])
    eval_parser.add_argument("--out", type=str, default="reports/sensitivity.json")
    
    args = parser.parse_args()
    
    if args.command == "train":
        from config import Config
        from env import AttackerType
        from train import train
        
        cfg = Config(
            N=args.N,
            B=args.B,
            total_steps=args.total_steps,
            seed=args.seed,
            runs_dir=args.runs_dir
        )
        
        attacker_map = {
            "static": AttackerType.STATIC,
            "sequential": AttackerType.SEQUENTIAL,
            "dynamic": AttackerType.DYNAMIC
        }
        
        train(cfg, attacker_map[args.attacker])
    
    elif args.command == "eval":
        from env import AttackerType
        from eval_sensitivity import evaluate_sensitivity
        
        attacker_map = {
            "static": AttackerType.STATIC,
            "sequential": AttackerType.SEQUENTIAL,
            "dynamic": AttackerType.DYNAMIC
        }
        
        evaluate_sensitivity(
            args.ckpt,
            num_samples=args.samples,
            attacker_type=attacker_map[args.attacker],
            output_path=args.out
        )
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
