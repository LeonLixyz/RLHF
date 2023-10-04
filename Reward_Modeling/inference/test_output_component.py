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
    enn_test_logger(reward_model, test_dataloader, device)

    reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, reward_diff_list, model_out_diff_list, label_list = test_logger(reward_model, test_dataloader, device)
    plot_reward_diff(reward_diff_list, model_out_diff_list, label_list, config)
    plot_components(reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, label_list, config)

def main():
    config = load_config("configs/test_config.json")
    test_dataloader = load_test_data(config, tokenizer)

    reward_model, tokenizer = load_trained_reward(config)

    minitest(reward_model, test_dataloader, config)

if __name__ == "__main__":
    main()
