import os
import json
import utils
import yaml


def get_args_str(hyperparameters):
    args_str = ''
    for arg, value in hyperparameters.items():
        args_str += f' --{arg} {value}'
    return args_str

def main():
    hyperparameters = utils.load_config("configs/reward_enn.json")
    gradient_acc = hyperparameters['gradient_acc']
    
    with open('train/accelerate_configs/DeepSpeed_s2.yaml', 'r') as f:
        deepspeed_config = yaml.safe_load(f)
    deepspeed_config['deepspeed_config']['gradient_accumulation_steps'] = gradient_acc
    with open('train/accelerate_configs/DeepSpeed_s2.yaml', 'w') as f:
        yaml.safe_dump(deepspeed_config, f)
    
    command = f'accelerate launch --config_file train/accelerate_configs/DeepSpeed_s2.yaml train/supervised_train.py {get_args_str(hyperparameters)}'
    os.system(command)

if __name__ == "__main__":
    main()