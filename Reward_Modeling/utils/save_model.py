import torch
import os

def save_model(logging_args, model):
    model.save_pretrained(logging_args.save_dir)