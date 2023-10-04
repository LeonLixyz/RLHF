import torch
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from alpaca_farm import common
import numpy as np
from models import AlpacaENN


def main():
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    config = utils.load_config("configs/vanilla_reward.json") # replace stuff in here
    model_dir = "/shared/share_mala/leon/reward-alpaca-enn/human"  
    print(config["pretrained_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["pretrained_model"],
        padding_side="left",
        )
    print('tokenizer:', tokenizer)

    # maybe use gpt-4 dataset here?
    #training_dataloader, eval_dataloader = utils.load_train_data(config, tokenizer)

    reward_alpaca = AlpacaENN.from_pretrained(
        model_dir,
        flash_attn=True,
        ref_size = 30,
        enn_hidden_size = 64,
        enn_output_size = 1,
        enn_gain = 0.8,
        lmbda = 0.3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_alpaca.to(device)

    print(reward_alpaca.reward_head.weight)

if __name__ == "__main__":
    main()

