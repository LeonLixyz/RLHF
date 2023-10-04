import torch
import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from alpaca_farm import common
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def distributed_evaluation(model, eval_dataloader, config):
    TP = 0    
    TN = 0    
    FP = 0    
    FN = 0   

    r_win_list = []
    r_lose_list = []

    with torch.no_grad():
        for _, test_batch in enumerate(eval_dataloader):

            if config["project"] == "vanilla-reward":
                p_1, p_2, p_1_att, p_2_att, label = test_batch
                p_1 = p_1.to(device)
                p_2 = p_2.to(device)
                p_1_att = p_1_att.to(device)
                p_2_att = p_2_att.to(device)
                label = torch.tensor(label).to(device)
                predictions, labels, eval_loss, r_win, r_loss = model.evaluate(p_1, p_2, p_1_att, p_2_att, label)
                # print('predictions', predictions)

            print('predictions', predictions)
            print('labels', labels)
            print('r_win', r_win)
            print('r_loss', r_loss)

            r_win_list.extend(r_win.tolist())
            r_lose_list.extend(r_loss.tolist())

            TP += ((predictions == 1) & (labels == 1)).sum().item()
            TN += ((predictions == 2) & (labels == 2)).sum().item()
            FP += ((predictions == 1) & (labels == 2)).sum().item()
            FN += ((predictions == 2) & (labels == 1)).sum().item()

    total_predictions = TP + TN + FP + FN
    total_actual_1 = TP + FN
    total_actual_2 = TN + FP
    total_predicted_1 = TP + FP
    total_predicted_2 = TN + FN


    accuracy = (TP + TN) / total_predictions
    label_1_rate = total_actual_1 / total_predictions
    label_2_rate = total_actual_2 / total_predictions
    prediction_1_rate = total_predicted_1 / total_predictions
    prediction_2_rate = total_predicted_2 / total_predictions
    true_1_rate = TP / total_actual_1
    true_2_rate = TN / total_actual_2

    r_win_list = torch.tensor(r_win_list)
    r_lose_list = torch.tensor(r_lose_list)
    r_total_list = torch.cat((r_win_list, r_lose_list), 0)

    r_win_average = r_win_list.mean()
    r_win_min = r_win_list.min()
    r_win_max = r_win_list.max()
    r_win_std = r_win_list.std()

    r_lose_average = r_lose_list.mean()
    r_lose_min = r_lose_list.min()
    r_lose_max = r_lose_list.max()
    r_lose_std = r_lose_list.std()

    r_total_average = r_total_list.mean()
    r_total_min = r_total_list.min()
    r_total_max = r_total_list.max()
    r_total_std = r_total_list.std()

    # return performance metrics and reward metrics
    metric = {
        "accuracy": accuracy,
        "label_1_rate": label_1_rate,
        "label_2_rate": label_2_rate,
        "prediction_1_rate": prediction_1_rate,
        "prediction_2_rate": prediction_2_rate,
        "true_1_rate": true_1_rate,
        "true_2_rate": true_2_rate,
        "eval_loss": eval_loss
    }
    reward = {
        "r_win_average": r_win_average,
        "r_win_min": r_win_min,
        "r_win_max": r_win_max,
        "r_win_std": r_win_std,
        "r_lose_average": r_lose_average,
        "r_lose_min": r_lose_min,
        "r_lose_max": r_lose_max,
        "r_lose_std": r_lose_std,
        "r_total_average": r_total_average,
        "r_total_min": r_total_min,
        "r_total_max": r_total_max,
        "r_total_std": r_total_std
    }

    return metric, reward

def minitest(model, eval_dataloader, config):


    model.eval()
    metrics, reward = distributed_evaluation(model, eval_dataloader, config)

    def print_metrics(metrics, category="metric"):
        if category == "metric":
            print(f'eval_loss: {metrics["eval_loss"]:.4f}, accuracy: {metrics["accuracy"]:.4f}')
            print(f'label_1_rate: {metrics["label_1_rate"]:.4f}, label_2_rate: {metrics["label_2_rate"]:.4f}, prediction_1_rate: {metrics["prediction_1_rate"]:.4f}, prediction_2_rate: {metrics["prediction_2_rate"]:.4f}')
            print(f'true_1_rate: {metrics["true_1_rate"]:.4f}, true_2_rate: {metrics["true_2_rate"]:.4f}')
        elif category == "reward":
            print(f'r_win_average: {metrics["r_win_average"]:.4f}, r_win_min: {metrics["r_win_min"]:.4f}, r_win_max: {metrics["r_win_max"]:.4f}, r_win_std: {metrics["r_win_std"]:.4f}')
            print(f'r_lose_average: {metrics["r_lose_average"]:.4f}, r_lose_min: {metrics["r_lose_min"]:.4f}, r_lose_max: {metrics["r_lose_max"]:.4f}, r_lose_std: {metrics["r_lose_std"]:.4f}')
            print(f'r_total_average: {metrics["r_total_average"]:.4f}, r_total_min: {metrics["r_total_min"]:.4f}, r_total_max: {metrics["r_total_max"]:.4f}, r_total_std: {metrics["r_total_std"]:.4f}')
            print('------------------------------------------------------------------------------------------')

    print_metrics(metrics, "metric")
    print_metrics(reward, "reward")
    #enn_test_logger(reward_model, test_dataloader, device)
    #reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, reward_diff_list, model_out_diff_list, label_list = test_logger(reward_model, test_dataloader, device)
    #plot_reward_diff(reward_diff_list, model_out_diff_list, label_list, config)
    #plot_components(reward_1_list, reward_2_list, model_out_1_list, model_out_2_list, Eta_out_1_list, Eta_out_2_list, P_out_1_list, P_out_2_list, label_list, config)

def main():
    config = utils.load_config("configs/vanilla_reward.json")
    model_dir = "/shared/share_mala/leon/reward-model-human"  # Replace with the actual path where you saved the models
    tokenizer_dir = "/shared/share_mala/leon/sft10k"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        padding_side="left",
        )
    print('tokenizer:', tokenizer)
    training_dataloader, eval_dataloader = utils.load_train_data(config, tokenizer)

    reward_alpaca = utils.AlpacaReward.from_pretrained(
        model_dir,
        flash_attn=True,
        bf16=True,
    )
    reward_alpaca.to(device)

    print(reward_alpaca.reward_head.weight)

    minitest(reward_alpaca, eval_dataloader, config)

if __name__ == "__main__":
    main()
