import os
from collections import defaultdict
import torch
from accelerate.utils import broadcast
from data import gen_ref


def broadcast_samples(num_ref, ref_size, accelerator):

    device = accelerator.device  
    
    if accelerator.is_main_process:
        z_samples = gen_ref(num_ref, ref_size).to(device)
    else:
        z_samples = torch.empty(num_ref, ref_size).to(device)
    
    broadcast(z_samples)
    return z_samples

def compute_loss_enn(model, p_1_ids, p_1_att, z_samples, label):
    r_1 = model(input_ids = p_1_ids, attention_mask = p_1_att, z_samples = z_samples, return_dict = True).rewards
    loss = torch.nn.functional.binary_cross_entropy_with_logits(r_1, label)
    loss = loss.mean()
    
    return loss   

def compute_loss_vanilla(model, p_1_ids, p_1_att, label):
    r_1 = model(input_ids = p_1_ids, attention_mask = p_1_att, return_dict = True).rewards
    loss = torch.nn.functional.binary_cross_entropy_with_logits(r_1, label)
    loss = loss.mean()
    
    return loss   

def supervised_trainer(model, optimizer, scheduler, training_dataloader, eval_dataloader, joint_eval_dataloader, accelerator, logging_args, training_args, model_args):

    from .supervised_eval import supervised_distributed_evaluation, supervised_joint_distributed_eval

    def print_metrics(accelerator, metrics, category="metric"):
        if category == "metric":
            accelerator.print(f'eval_loss: {metrics["eval_loss"]:.4f}, accuracy: {metrics["accuracy"]:.4f}')
            accelerator.print(f'label_1_rate: {metrics["label_1_rate"]:.4f}, label_2_rate: {metrics["label_2_rate"]:.4f}, prediction_1_rate: {metrics["prediction_1_rate"]:.4f}, prediction_2_rate: {metrics["prediction_2_rate"]:.4f}')
            accelerator.print(f'true_1_rate: {metrics["true_1_rate"]:.4f}, true_2_rate: {metrics["true_2_rate"]:.4f}')
        elif category == "reward":
            accelerator.print(f'r_win_average: {metrics["r_win_average"]:.4f}, r_win_min: {metrics["r_win_min"]:.4f}, r_win_max: {metrics["r_win_max"]:.4f}, r_win_std: {metrics["r_win_std"]:.4f}')
            if logging_args.project == "reward-enn":
                accelerator.print(f'eta_win_average: {metrics["eta_win_average"]:.4f}, eta_win_min: {metrics["eta_win_min"]:.4f}, eta_win_max: {metrics["eta_win_max"]:.4f}, eta_win_std: {metrics["eta_win_std"]:.4f}')
                accelerator.print(f'p_win_average: {metrics["p_win_average"]:.4f}, p_win_min: {metrics["p_win_min"]:.4f}, p_win_max: {metrics["p_win_max"]:.4f}, p_win_std: {metrics["p_win_std"]:.4f}')
            
    def eval_logger(accelerator, metrics_storage, rewards_storage):
        if logging_args.project == "reward-enn":
            eval_z_size_list = training_args.eval_z_size_list
            for eval_z_samples_size in eval_z_size_list:
                metrics, reward = supervised_distributed_evaluation(model, eval_dataloader, accelerator, model_args, logging_args, training_args, eval_z_samples_size)
                log_joint_likelihood, joint_log_likelihood = supervised_joint_distributed_eval(model, joint_eval_dataloader, training_args, logging_args, model_args, accelerator, eval_z_samples_size)
                accelerator.print('eval_z_samples_size:', eval_z_samples_size)
                print_metrics(accelerator, metrics, "metric")
                accelerator.print('log joint likelihood:', log_joint_likelihood, 'joint log likelihood:', joint_log_likelihood)
                accelerator.log({f"z_{eval_z_samples_size}_joint_log_likelihood": log_joint_likelihood})
                accelerator.log({f"z_{eval_z_samples_size}_log_joint_likelihood": joint_log_likelihood})               
                print_metrics(accelerator, reward, "reward")
                accelerator.print('')            

                metrics_storage['accuracy'].append(metrics['accuracy'])
                metrics_storage['eval_loss'].append(metrics['eval_loss'])

                for key in reward:
                    rewards_storage[key].append(reward[key])

                prefix = f'z_{eval_z_samples_size}_'
                accelerator.log({prefix + key: value for key, value in metrics.items()})
                accelerator.log({prefix + key: value for key, value in reward.items()})

        elif logging_args.project == "vanilla-reward":
            metrics, reward = supervised_distributed_evaluation(model, eval_dataloader, accelerator, model_args, logging_args, training_args)
            log_joint_likelihood, joint_log_likelihood = supervised_joint_distributed_eval(model, joint_eval_dataloader, training_args, logging_args, model_args, accelerator)
            print_metrics(accelerator, metrics, "metric")
            accelerator.print('log joint likelihood:', log_joint_likelihood, 'joint log likelihood:', joint_log_likelihood)
            accelerator.log({"joint_log_likelihood": log_joint_likelihood})
            accelerator.log({"log_joint_likelihood": joint_log_likelihood})
            print_metrics(accelerator, reward, "reward")
            accelerator.print('')            

            metrics_storage['accuracy'].append(metrics['accuracy'])
            metrics_storage['eval_loss'].append(metrics['eval_loss'])

            for key in reward:
                rewards_storage[key].append(reward[key])

            accelerator.log(metrics)
            accelerator.log(reward)

        accelerator.print('------------------------------------------------------------------------------------------')


    wandb_dir = logging_args.wandb_dir
    wandb_run_name = logging_args.wandb_run_name
        
    if accelerator.is_main_process:
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)

    # Initialize wandb

    accelerator.init_trackers(
        project_name =logging_args.wandb_project,
        init_kwargs={"wandb": {"name": wandb_run_name, 
                               "dir": wandb_dir}}
        )

    eval_steps = training_args.eval_steps
    metrics_storage = defaultdict(list)
    rewards_storage = defaultdict(list)

    current_z_samples = None

    for e in range(training_args.num_epochs):
        accelerator.print('epoch', e)
        # brefore training eval:
        accelerator.print('eval before training')
        eval_logger(accelerator, metrics_storage, rewards_storage)

        for i, batch in enumerate(training_dataloader):

            optimizer.zero_grad()
            p_1_ids, p_1_att, label = batch

            if logging_args.project == "reward-enn":

                if i % training_args.gradient_acc == 0:
                    current_z_samples = broadcast_samples(model_args.num_ref_train, model_args.ref_size, accelerator)
                    if training_args.bf16:
                        current_z_samples = current_z_samples.to(torch.bfloat16)
                    elif training_args.fp16:
                        current_z_samples = current_z_samples.to(torch.half)

                loss = compute_loss_enn(model = model, p_1_ids = p_1_ids, p_1_att = p_1_att, z_samples = current_z_samples, label = label)

            elif logging_args.project == "vanilla-reward":
                loss = compute_loss_vanilla(model = model, p_1_ids = p_1_ids, p_1_att = p_1_att, label = label)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            gathered_loss = accelerator.gather(loss)
            metrics_storage["training_loss"].append(gathered_loss.mean().item())
            accelerator.log({"training_loss": gathered_loss.mean().item()})

            if (i+1) % eval_steps == 0:
                # Evaluation    
                accelerator.print(f'epoch {e+1} step {i+1} evaluation')
                eval_logger(accelerator, metrics_storage, rewards_storage)
    

        # end of epoch eval:
        accelerator.print(f'epoch {e+1} evaluation')
        eval_logger(accelerator, metrics_storage, rewards_storage)

    accelerator.end_training()


    return model, metrics_storage, rewards_storage



