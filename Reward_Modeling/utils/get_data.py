import json
import random
from data.data_loader import pairwise_data_tokenized, PairwiseDyadicAugmentedTokenizedData, single_data_tokenized, SingleDyadicAugmentedTokenizedData 
import torch


def load_train_data(data_args, tokenizer):
    data_dir = data_args.data_dir
    train_dir = data_dir + "/train.json"
    eval_dir = data_dir + "/eval.json"
    joint_eval_dir = data_dir + "/unaug_joint_eval.json"
    prompts_dir = data_dir + "/prompts.json"
    with open(train_dir) as f:
        train_raw_data = json.load(f)
    with open(eval_dir) as f:
        eval_raw_data = json.load(f)
    with open(joint_eval_dir) as f:
        joint_eval_raw_data = json.load(f)
    with open(prompts_dir) as f:
        prompts = json.load(f)

    model_max_length = data_args.max_length

    # round down to the nearst multiple of 8: 
    train_raw_data = train_raw_data[:len(train_raw_data) - len(train_raw_data) % (8 * data_args.train_batch_size) ]
    train_dataset = pairwise_data_tokenized(train_raw_data, tokenizer, model_max_length, prompts)
    eval_dataset = pairwise_data_tokenized(eval_raw_data, tokenizer, model_max_length, prompts)
    joint_eval_dataset = PairwiseDyadicAugmentedTokenizedData(joint_eval_raw_data, tokenizer, model_max_length, prompts)
    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=data_args.train_batch_size, shuffle = True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=data_args.eval_batch_size, shuffle= False)
    joint_eval_dataloader = torch.utils.data.DataLoader(joint_eval_dataset, batch_size=1, shuffle= False)

    return training_dataloader, eval_dataloader, joint_eval_dataloader

def load_supervised_data(data_args, tokenizer):

    data_dir = data_args.data_dir
    train_dir = data_dir + "/train.json"
    eval_dir = data_dir + "/eval.json"
    joint_eval_dir = data_dir + "/unaug_joint_eval.json"
    prompts_dir = data_dir + "/prompts.json"
    with open(train_dir) as f:
        train_raw_data = json.load(f)
    with open(eval_dir) as f:
        eval_raw_data = json.load(f)
    with open(joint_eval_dir) as f:
        joint_eval_raw_data = json.load(f)
    with open(prompts_dir) as f:
        prompts = json.load(f)

    model_max_length = data_args.max_length

    train_raw_data = train_raw_data[:len(train_raw_data) - len(train_raw_data) % (8 * data_args.train_batch_size)]
    train_dataset = single_data_tokenized(train_raw_data, tokenizer, model_max_length, prompts)
    eval_dataset = single_data_tokenized(eval_raw_data, tokenizer, model_max_length, prompts)
    joint_eval_dataset = SingleDyadicAugmentedTokenizedData(joint_eval_raw_data, tokenizer, model_max_length, prompts)
    training_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=data_args.train_batch_size, shuffle = True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=data_args.eval_batch_size, shuffle= False)
    joint_eval_dataloader = torch.utils.data.DataLoader(joint_eval_dataset, batch_size=1, shuffle= False)

    return training_dataloader, eval_dataloader, joint_eval_dataloader


def load_test_data(config, tokenizer):
    test_data_dir = config['data_dir'].format(config['test_dataset_name'] + "_eval")
    with open(test_data_dir) as f:
        annotated = json.load(f)[:100]

    test_dataset = pairwise_data_tokenized(annotated, tokenizer, config)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["eval_batch_size"], shuffle=False)


    return test_dataloader