import torch
from utils import load_config, load_trained_reward, load_test_data, plot_reward_diff,  plot_components, test_logger, enn_test_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from alpaca_farm import common
import matplotlib.pyplot as plt
import numpy as np

def minitest(reward_model, test_dataloader, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    reward_model.to(device)
    instruction = "Describe the responsibilities of the given job. Security Officer"
    Response_1 = "A Security Officer is responsible for maintaining security at a given location. This may include conducting security checks, patrolling the premises, monitoring surveillance systems, and responding to alarms. The Security Officer is also tasked with enforcing rules and regulations, intervening in situations, and documenting security-related incidents. Additionally, the Security Officer may be required to educate staff on security protocols and provide security-related training."
    Response_2 = "I don't know, I don't know"
    prompts = {
        "prompt_noinputs": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "prompt_inputs": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    }
    

def main():
    config = load_config("configs/test_config.json")

    model_dir = config['save_dir'].format(
            config['user'],
            config['project'],
            config['dataset_name'],
            config['scheduler_type'],
            config['lr'],
            config['warmup_ratio'],
            config['batch_size'],
            config['ref_size'],
            config['num_ref'],
            config['hidden_size'],
            config['enn_gain'])

    mix_precision = True
    base_model = AutoModelForCausalLM.from_pretrained(model_dir + "/base_model")
    base_model = base_model.to(dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

    test_dataloader = load_test_data(config, tokenizer)

    reward_model = load_trained_reward(config, base_model, mix_precision)

    minitest(reward_model, test_dataloader, config)

if __name__ == "__main__":
    main()

