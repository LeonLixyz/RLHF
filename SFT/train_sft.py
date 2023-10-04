import os
import json
import yaml
def load_config(config_path):
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    return config


def get_args_str(hyperparameters):
    args_str = ''
    for arg, value in hyperparameters.items():
        args_str += f' --{arg} {value}'
    return args_str

def main():
    hyperparameters = load_config("sft.json")
    gradient_acc = hyperparameters['gradient_accumulation_steps']
    
    with open('DeepSpeed_sft.yaml', 'r') as f:
        deepspeed_config = yaml.safe_load(f)
    print('deepspeed_config',deepspeed_config['deepspeed_config'])
    deepspeed_config['deepspeed_config']['gradient_accumulation_steps'] = gradient_acc
    with open('DeepSpeed_sft.yaml', 'w') as f:
        yaml.safe_dump(deepspeed_config, f)
    
    command = f'accelerate launch --config_file DeepSpeed_sft.yaml sft.py {get_args_str(hyperparameters)}'
    os.system(command)

if __name__ == "__main__":
    main()
    