from dataclasses import dataclass, field
from typing import Optional
import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import trl

# Define and parse arguments.
@dataclass
class ModelArguments:
    model_name: str = field(default = "EleutherAI/pythia-2.8b-v0")
    bf16: bool = field(default = True)
    max_seq_length: int = field(default = 512)

@dataclass
class DataArguments:
    dataset_name: str = field(default = "tatsu-lab/alpaca_farm")
    dataset_split: str = field(default = "sft")
    batch_size: int = field(default = 2)


@dataclass 
class TrainArguments:
    seed: int = field(default = 42)
    output_dir: str = field(default = "/shared/share_mala/leon/pythia-2.8b-sft")
    gradient_accumulation_steps: int = field(default = 16)
    learning_rate: float = field(default = 2e-5)
    logging_steps: int = field(default = 1)
    num_train_epochs: int = field(default = 3)
    log_with: str = field(default = "wandb")
    warmup_ratio: float = field(default = 0.03)
    weight_decay: float = field(default = 0.01)


def main():
    parser = HfArgumentParser((ModelArguments, TrainArguments, DataArguments))
    model_args, train_args, data_args = parser.parse_args_into_dataclasses()
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'

    trl.set_seed(train_args.seed)

    tqdm.pandas()

    dataset = load_dataset(data_args.dataset_name, split=data_args.dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    if model_args.bf16:
        torch_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, torch_dtype=torch_dtype, device_map='auto',
    )
    else:
        torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, torch_dtype=torch_dtype, device_map='auto',
    )

    def formatting_prompts_func(example):
        prompts = {
            "prompt_noinputs": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "prompt_inputs": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        }

        output_texts = []

        for i in range(len(example['instruction'])):
            if example['input'][i] == '':
                text = prompts['prompt_noinputs'].format(instruction=example['instruction'][i]) + example['output'][i]
            else:
                text = prompts['prompt_inputs'].format(instruction=example['instruction'][i], input=example['input'][i]) + example['output'][i]
            
            output_texts.append(text)

        return output_texts
    

    response_template = "\n### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        per_device_train_batch_size=data_args.batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        learning_rate=train_args.learning_rate,
        logging_steps=train_args.logging_steps,
        num_train_epochs=train_args.num_train_epochs,
        report_to=train_args.log_with,
        bf16 = model_args.bf16,
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=model_args.max_seq_length,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train() 

    trainer.save_model(train_args.output_dir)

if __name__ == "__main__":
    main()