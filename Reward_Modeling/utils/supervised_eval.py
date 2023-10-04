import torch
import time
from tqdm import tqdm
from .get_trainer import broadcast_samples, gen_ref

def gather_and_convert_to_list(accelerator, tensor_dict, key):
    """Helper function to gather tensor from accelerator and convert to list."""
    tensor = accelerator.gather(tensor_dict[key]).cpu()
    return tensor.tolist()

def compute_stats(tensor_list):
    """Compute statistical metrics for a list of tensors."""
    tensor = torch.tensor(tensor_list)
    return {
        "average": tensor.mean().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "std": tensor.std().item()
    }

def compute_eval_vanilla(model, p_1_ids, p_1_att, label):
    r_1 = model(input_ids = p_1_ids, attention_mask = p_1_att, return_dict = True).rewards
    eval_loss = torch.nn.functional.binary_cross_entropy_with_logits(r_1, label)
    eval_loss = eval_loss.mean()

    predictions = (torch.sigmoid(r_1) > 0.5).float().squeeze()

    #return a disctionary of metrics
    results = {
        "predictions": predictions,
        "label": label.squeeze(-1).long(),
        "eval_loss": eval_loss,
        "r_win": r_1,
    }
    return results
    
def compute_eval_enn(model, p_1_ids, p_1_att, z_samples, label):
    ENNComponet_1 = model(input_ids = p_1_ids, attention_mask = p_1_att, z_samples = z_samples, return_full_z = True, return_dict = True)

    # compute model out
    model_out_1 = ENNComponet_1.model_out
    
    # Compute reward diff
    r_1 = torch.mean(ENNComponet_1.reward_list, dim=1).view(-1, 1)
    eta_1 = torch.mean(ENNComponet_1.eta_list, dim=1).view(-1, 1)
    p_1 = torch.mean(ENNComponet_1.p_list, dim=1).view(-1, 1)

    eval_loss = torch.nn.functional.binary_cross_entropy_with_logits(r_1, label)
    eval_loss = eval_loss.mean()

    predictions = (torch.sigmoid(r_1) > 0.5).float().squeeze()

    # Returning results as a dictionary
    results = {
        "predictions": predictions,
        "label": label.squeeze(-1).long(),
        "eval_loss": eval_loss,
        "model_out_win": model_out_1,
        "r_win": r_1,
        'eta_win': eta_1,
        'p_win': p_1,
    }

    return results

def supervised_distributed_evaluation(model, eval_dataloader, accelerator, model_args, logging_args, training_args, eval_z_samples_size = None):
    TP, TN, FP, FN = 0, 0, 0, 0
    metrics = ["r_win"]
    eval_loss_list = []
    if logging_args.project == "reward-enn":
        metrics.extend(["model_out_win", "eta_win", "p_win"])
        eval_z_samples = broadcast_samples(eval_z_samples_size, model_args.ref_size, accelerator)

        if training_args.bf16:
            eval_z_samples = eval_z_samples.to(torch.bfloat16)
        elif training_args.fp16:
            eval_z_samples = eval_z_samples.to(torch.half)
    
    results_list = {metric: [] for metric in metrics}
    
    with torch.no_grad():
        for _, test_batch in enumerate(tqdm(eval_dataloader)):
            p_1_ids, p_1_att, label = test_batch
            eval_params = {
                "model": model,
                "p_1_ids": p_1_ids, 
                "p_1_att": p_1_att, 
                "label": label
            }
            
            if logging_args.project == "reward-enn":
                eval_params["z_samples"] = eval_z_samples
                results = compute_eval_enn(**eval_params)
            elif logging_args.project == "vanilla-reward":
                results = compute_eval_vanilla(**eval_params)

            # Update results

            for metric in metrics:
                results_list[metric].extend(gather_and_convert_to_list(accelerator, results, metric))

            predictions = accelerator.gather(results['predictions']).cpu()
            labels = accelerator.gather(results['label']).cpu()
            batch_eval_loss = accelerator.gather(results['eval_loss']).cpu()
            eval_loss_list.extend(batch_eval_loss.tolist())

            TP += ((predictions == 1) & (labels == 1)).sum().item()
            TN += ((predictions == 0) & (labels == 0)).sum().item()
            FP += ((predictions == 1) & (labels == 0)).sum().item()
            FN += ((predictions == 0) & (labels == 1)).sum().item()

    # Compute metrics
    total_predictions = TP + TN + FP + FN
    total_actual_1 = TP + FN
    total_actual_2 = TN + FP


    metric = {
        "accuracy": (TP + TN) / total_predictions,
        "label_1_rate": total_actual_1 / total_predictions,
        "label_2_rate": total_actual_2 / total_predictions,
        "prediction_1_rate": (TP + FP) / total_predictions,
        "prediction_2_rate": (TN + FN) / total_predictions,
        "true_1_rate": TP / total_actual_1,
        "true_2_rate": TN / total_actual_2,
        "eval_loss": torch.tensor(eval_loss_list).mean().item()
    }

    # Compute reward stats
    reward = {}
    for metric_key in metrics:
        stats = compute_stats(results_list[metric_key])
        for key, val in stats.items():
            reward[f"{metric_key}_{key}"] = val

    return metric, reward


def supervised_joint_distributed_eval(model, joint_eval_dataloader, training_args, logging_args, model_args, accelerator, eval_z_samples_size = None):

    joint_log_list = []
    log_joint_list = []
    joint_loss_list = []
    with torch.no_grad():
        for i, batch in enumerate(joint_eval_dataloader):

            p_1_ids, p_1_att, labels = batch
            p_1_ids = p_1_ids.squeeze()
            p_1_att = p_1_att.squeeze()
            labels = labels.squeeze()

            if logging_args.project == "reward-enn":
                z_samples = gen_ref(eval_z_samples_size, model_args.ref_size).to(accelerator.device)
                if training_args.bf16:
                    z_samples = z_samples.to(torch.bfloat16)
                elif training_args.fp16:
                    z_samples = z_samples.to(torch.half)

                reward_list_1 = model(p_1_ids, attention_mask=p_1_att, z_samples=z_samples, return_full_z=True).reward_list 

            elif logging_args.project == "vanilla-reward":

                reward_list_1 = model(p_1_ids, attention_mask=p_1_att, return_dict=True).rewards
      
            #accelerator.print('reward_list_1', reward_list_1, reward_list_1.shape)
            #accelerator.print('labels', labels, labels.shape)

            sigmoid_values = torch.sigmoid(reward_list_1)
            #accelerator.print('sigmoid', sigmoid_values, sigmoid_values.shape)
            likelihood_list = torch.where(labels.unsqueeze(-1) == 1, sigmoid_values, 1 - sigmoid_values)
            #accelerator.print('likelihood', likelihood_list, likelihood_list.shape)
            likelihood_list = torch.clamp(likelihood_list, min=1e-5, max=1-1e-5)
            #accelerator.print('likelihood', likelihood_list)
            # joint log likelihood
            log_softmax_values = torch.log(likelihood_list)
            #accelerator.print('log_softmax_values', log_softmax_values)
            #accelerator.print('log sigmoid', log_softmax_values)
            # averaging over the z samples, if vanilla then dimension is 1 so no effect
            log_softmax_values = torch.mean(log_softmax_values, dim=1)
            #accelerator.print('mean', log_softmax_values)
            # summing all dyadic samples
            joint_log_likelihood = torch.sum(log_softmax_values, dim=0)
            #accelerator.print('sum', joint_log_likelihood)
            joint_log_likelihood_gathered = accelerator.gather(joint_log_likelihood).cpu()
            joint_log_list.extend(joint_log_likelihood_gathered.tolist())

            product_softmax_values = torch.prod(likelihood_list, dim=0)
            #accelerator.print('prod', product_softmax_values.shape)
            # Averaging over the z samples, if vanilla then dimension is 1 so no effect
            avg_softmax_values = torch.mean(product_softmax_values)
            #accelerator.print('mean', avg_softmax_values.shape)
            # log joint likelihood
            log_joint_likelihood = torch.log(avg_softmax_values)
            #accelerator.print('log', log_joint_likelihood, log_joint_likelihood.shape)
            log_joint_likelihood_gathered = accelerator.gather(log_joint_likelihood).cpu()
            log_joint_list.extend(log_joint_likelihood_gathered.tolist())
        
        log_joint_likelihood = torch.tensor(log_joint_list).sum()
        joint_log_likelihood = torch.tensor(joint_log_list).sum()

        return log_joint_likelihood, joint_log_likelihood
