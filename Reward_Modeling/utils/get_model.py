
from models.enn_net import Epinet
from models.reward_enn import RewardENN, RewardENNConfig
from models.vanilla_reward import VanillaReward, VanillaRewardConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# create_new_model_and_optimizer
def create_new_model_and_optimizer(model_args, training_args, logging_args):

    if logging_args.project == "reward-enn":
        config = RewardENNConfig(
            backbone_model_name_or_path=model_args.backbone_model,
            ref_size=model_args.ref_size,
            enn_hidden_size=model_args.hidden_size,
            enn_output_size=model_args.output_size,
            enn_gain=model_args.enn_gain,
            lmbda=model_args.lmbda,
            )
        model = RewardENN(
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=config,
        )

        optimizer = torch. optim.AdamW([
                    {'params': model.backbone_model.parameters(), 'lr': training_args.lr, 'weight_decay': training_args.weight_decay},  
                    {'params': model.reward_head.parameters(), 'lr': training_args.reward_lr, 'weight_decay': training_args.reward_decay},
                    {'params': model.eta_net.parameters(), 'lr': training_args.enn_lr, 'weight_decay': training_args.enn_decay} 
                ])

    elif logging_args.project == "vanilla-reward":
        config = VanillaRewardConfig(
            backbone_model_name_or_path=model_args.backbone_model,
            reward_gain=model_args.reward_gain,
            )
        
        model = VanillaReward(
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=config,
        )

        optimizer = torch. optim.AdamW([
                    {'params': model.backbone_model.parameters(), 'lr': training_args.lr, 'weight_decay': training_args.weight_decay},  
                    {'params': model.reward_head.parameters(), 'lr': training_args.reward_lr, 'weight_decay': training_args.reward_decay},
                ])

    return model, optimizer

def load_trained_reward(model_args, training_args, logging_args):

    mix_precision = True

    model_dir = logging_args.save_dir
    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

    if logging_args.project == "reward-enn":
        config = RewardENNConfig(
            backbone_model_name_or_path=model_args.backbone_model,
            ref_size=model_args.ref_size,
            enn_hidden_size=model_args.hidden_size,
            enn_output_size=model_args.output_size,
            enn_gain=model_args.enn_gain,
            lmbda=model_args.lmbda,
            )
        model = RewardENN.from_pretrained(
            model_dir,
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=config,
            )


    elif logging_args.project == "vanilla-reward":
        config = VanillaRewardConfig(
            backbone_model_name_or_path=model_args.backbone_model,
            reward_gain=model_args.reward_gain,
            )
        
        model = VanillaReward.from_pretrained(
            model_dir,
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=config,
        )




    return tokenizer, model