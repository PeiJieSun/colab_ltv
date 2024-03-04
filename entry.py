import argparse
import yaml
import os, sys
import subprocess

import readline, GPUtil

def custom_input(prompt):
    readline.set_startup_hook(lambda: readline.insert_text(prompt))
    try:
        user_input = input()
        return user_input[len(prompt):]
    finally:
        readline.set_startup_hook()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PMF Recommendation Model Example')
    parser.add_argument('--config', type=str, default='config/pmf.yaml', help='Config file path')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['model'] = args.model

    from train import train_model

    train_model(args, config)