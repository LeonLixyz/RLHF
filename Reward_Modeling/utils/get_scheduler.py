import transformers
import math
import torch

def get_scheduler(optimizer, training_args, data_args, training_dataloader, accelerator):
    num_epochs = training_args.num_epochs
    warmup_ratio = training_args.warmup_ratio
    total_data_samples = len(training_dataloader.dataset)

    num_training_steps = int(num_epochs * total_data_samples / (data_args.train_batch_size))
    accelerator.print('Total steps: ', num_training_steps)    
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    accelerator.print('Warmup steps: ', num_warmup_steps)

    def cosine_warm_up(warmup_steps, total_steps, warmup_factor=0.1, eta_min_ratio=0.1):
        def _lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = current_step / warmup_steps
                factor = warmup_factor * (1 - alpha) + alpha
                return factor
            
            # Progress after warmup
            alpha = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * alpha))
            return eta_min_ratio + (1 - eta_min_ratio) * cosine_decay
        
        return _lr_lambda

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warm_up(num_warmup_steps, num_training_steps))
 
    return scheduler
