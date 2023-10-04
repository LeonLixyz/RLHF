import json
import os
def load_config(config_path):
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    return config

def generate_save_dir(model_args, training_args, data_args, logging_args):
    # Extracting the required configs
    if logging_args.project == "reward-enn":
        config = {
            'user': logging_args.user,
            'project': logging_args.project,
            'dataset_name': data_args.dataset_name,
            'lr': training_args.lr,
            'ref_size': model_args.ref_size,
            'hidden_size': model_args.hidden_size,
            'gradient_acc': training_args.gradient_acc,
            'train_batch_size': data_args.train_batch_size,
            'num_ref_train': model_args.num_ref_train,
            'weight_decay': training_args.weight_decay,
            'wandb_project': logging_args.wandb_project,
            "enn_lr": training_args.enn_lr,
            "enn_decay": training_args.enn_decay,
            "reward_lr": training_args.reward_lr,
            "reward_decay": training_args.reward_decay,
            "num_epochs": training_args.num_epochs,
        }
    elif logging_args.project == "vanilla-reward":
        config = {
            'user': logging_args.user,
            'project': logging_args.project,
            'dataset_name': data_args.dataset_name,
            'lr': training_args.lr,
            'gradient_acc': training_args.gradient_acc,
            'train_batch_size': data_args.train_batch_size,
            'weight_decay': training_args.weight_decay,
            'wandb_project': logging_args.wandb_project

        }
    os.getcwd()
    data_args.data_dir = os.getcwd() + "/data/dataset/{}".format(data_args.dataset_name)
    # Define format strings based on project type
    if logging_args.project == "reward-enn":
        save_dir_format = "/shared/share_mala/{user}/{project}/{dataset_name}/-ref_size{ref_size}-enn_dim{hidden_size}-num_ref_train{num_ref_train}-lr{lr}-weight_decay{weight_decay}-enn_lr{enn_lr}-enn_decay{enn_decay}-reward_lr{reward_lr}-reward_decay{reward_decay}-gc{gradient_acc}-train_batch_size{train_batch_size}"
        plot_dir_format = "/shared/share_mala/{user}/Logs/plots/{project}/{dataset_name}/-ref_size{ref_size}-enn_dim{hidden_size}-num_ref_train{num_ref_train}-lr{lr}-weight_decay{weight_decay}-enn_lr{enn_lr}-enn_decay{enn_decay}-reward_lr{reward_lr}-reward_decay{reward_decay}-gc{gradient_acc}-train_batch_size{train_batch_size}"
        wandb_run_name_format = "{project}-{dataset_name}-ref_size{ref_size}-enn_dim{hidden_size}-num_ref_train{num_ref_train}-lr{lr}-weight_decay{weight_decay}-enn_lr{enn_lr}-enn_decay{enn_decay}-reward_lr{reward_lr}-reward_decay{reward_decay}-gc{gradient_acc}-train_batch_size{train_batch_size}"
        wandb_dir_format = "/shared/share_mala/{user}/Logs/wandb_logs/{project}/{wandb_project}/{dataset_name}-ref_size{ref_size}-enn_dim{hidden_size}-num_ref_train{num_ref_train}-lr{lr}-weight_decay{weight_decay}-enn_lr{enn_lr}-enn_decay{enn_decay}-reward_lr{reward_lr}-reward_decay{reward_decay}-gc{gradient_acc}-train_batch_size{train_batch_size}"
    
    elif logging_args.project == "vanilla-reward":
        save_dir_format = "/shared/share_mala/{user}/{project}/{dataset_name}-lr{lr}-gradient_acc{gradient_acc}-train_batch_size{train_batch_size}-weight_decay{weight_decay}"
        plot_dir_format = "/shared/share_mala/{user}/Logs/plots/{project}/{dataset_name}-lr{lr}-gradient_acc{gradient_acc}-train_batch_size{train_batch_size}-weight_decay{weight_decay}"
        wandb_run_name_format = "{project}-{dataset_name}-lr{lr}-gradient_acc{gradient_acc}-train_batch_size{train_batch_size}-weight_decay{weight_decay}"
        wandb_dir_format = "/shared/share_mala/{user}/Logs/wandb_logs/{project}/{wandb_project}/{dataset_name}-lr{lr}-gradient_acc{gradient_acc}-train_batch_size{train_batch_size}-weight_decay{weight_decay}"
    
    # Create or set the attributes with the appropriate formatted values
    logging_args.save_dir = save_dir_format.format(**config)
    logging_args.plot_dir = plot_dir_format.format(**config)
    logging_args.wandb_run_name = wandb_run_name_format.format(**config)
    logging_args.wandb_dir = wandb_dir_format.format(**config)