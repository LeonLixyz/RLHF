import os
import json
import itertools
import utils
import yaml

def get_args_str(hyperparameters):
    args_str = ''
    for arg, value in hyperparameters.items():
        args_str += f' --{arg} {value}'
    return args_str

def main():
    hyperparameters = utils.load_config("configs/reward_enn.json")


    hyperparameter_sweep_values = {
        "gradient_acc": [1],
        "train_batch_size": [8],
        "wandb_project": ["imdb"],
        "backbone_model": ["openlm-research/open_llama_3b_v2"],
        "dataset_name": ["IMDBDataset"],
        "ref_size": [10, 50, 100],
        "hidden_size": [64, 128],
        "num_ref_train": [100, 1000],
    }

    combinations = list(itertools.product(*hyperparameter_sweep_values.values()))

    for combo in combinations:
        for i, key in enumerate(hyperparameter_sweep_values.keys()):
            hyperparameters[key] = combo[i]

        print(hyperparameters)

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