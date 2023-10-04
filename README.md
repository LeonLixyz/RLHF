# Uncertainty Quantification for RLHF

This repository contains an implementation of the UQ for RLHF.

## Setup

1. Create a virtual environment:
   pip:

   ```shell
   python3 -m venv rlhf_env
   ```

   conda:
   ```shel
   conda activate rlhf_env
   ```

   

2. Activate the virtual environment:
   pip:

   ```shell
   source rlhf_env/bin/activate
   ```

   conda:
   ```shell
   conda activate rlhf_env
   ```

3. To install the correct torch and CUDA on the cluster, follow those steps:

   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
   ```

4. Install the required dependencies:
   You can try install with the requirement.txt I provided:

   ```shell
   pip install -r requirements.txt
   ```

   Or install the following packages:
   ```shell
   pip install transformer
   pip install accelerate
   pip install deepspeed
   pip install alpaca-farm
   ```

## Standard Finetuning

Our standard finetuning is based on Trlx and Alpaca Farm

## Reward Modeling

### Files

- `data`: Include `data_loader.py` and `dataset`
- `model`: `reward_model.py` and `enn_net.py`: implementation of reward_enn
- `train`: different methods of training, including `DeepSpeed`, `LoRA`, `FSDP`
- `inference`: load trained reward model and do inference and validation, has both standard trained and LoRA trained model. 
- `notebooks`: most of them are outdated and chaotic, use scripts instead. 
- `config`: folder of training configs of hyperparameters for training.
- `utils`: most of the training, validating, logging, and etc are here.

For reward enn modeling, run

```python
python -m scripts.train_reward_enn
```

To change hyperparameters, modify the hyperparameters in the `reward_enn.json` in the `configs` folder


For vanilla reward modelings, run

```python
python -m scripts.train_vanilla_reward
```

To change hyperparameters, modify the hyperparameters in the `vanilla_reward.json` in the `configs` folder


### Hyperparameter Tuning

The hyperparameter grid style sweeping pipeline is in `scripts/hp_sweep.py`. To conduct hyperparameter sweeping, run:

```python
python -m scripts.hp_sweep
```

### Inference

To load pretrained code and test for inference, try:

```python
python -m inference.reward_enn_inference
```

To enable `flash attention` for the base Llama model, uncomment the following lines in `inference/reward_enn_inference`

```python
mix_precision = True
base_model = common.make_generative_lm(config["save_dir"] + "/base_model", flash_attn = True, bf16 = True)
```

and comment the other two:
```python
mix_precision = False
base_model = AutoModelForCausalLM.from_pretrained(config["save_dir"] + "/base_model")
```

## 