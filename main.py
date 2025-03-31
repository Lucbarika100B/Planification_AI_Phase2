import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo", "hybrid"], required=True)
    args = parser.parse_args()

    if args.algo == "dqn":
        import src.train_dqn
    elif args.algo == "ppo":
        import src.train_ppo
    else:
        import src.hybrid_strategy
        