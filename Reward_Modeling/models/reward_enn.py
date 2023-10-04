import torch
import transformers
from models.enn_net import Epinet
from torch import Tensor, nn
from transformers.utils.generic import ModelOutput
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer



class RewardENNConfig(transformers.PretrainedConfig):
    model_type = "reward_enn"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(
                self, 
                backbone_model_name_or_path=None, 
                ref_size=None, 
                enn_hidden_size=None,
                enn_output_size=None,
                enn_gain=None,
                lmbda=None,
                **kwargs):
        super(RewardENNConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self._name_or_path = backbone_model_name_or_path
        self.ref_size = ref_size
        self.enn_hidden_size = enn_hidden_size
        self.enn_output_size = enn_output_size
        self.enn_gain = enn_gain
        self.lmbda = lmbda



class RewardModelOutput(ModelOutput):
    rewards: Tensor = None

class ENNComponet(ModelOutput):
    reward_list: Tensor = None
    eta_list: Tensor = None
    p_list: Tensor = None

def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers, "LLaMAForCausalLM" if hasattr(transformers, "LLaMAForCausalLM") else "LlamaForCausalLM"
        )
        if isinstance(model, llama_cls):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardENN(transformers.PreTrainedModel):
    config_class = RewardENNConfig

    def __init__(self, config: RewardENNConfig, **kwargs):
        super(RewardENN, self).__init__(config)
        self.ref_size = config.ref_size
        self.enn_hidden_size = config.enn_hidden_size
        self.enn_output_size = config.enn_output_size
        self.enn_gain = config.enn_gain
        self.lmbda = config.lmbda

        self.flash_attn = kwargs.get('flash_attn', None)
        if self.flash_attn:
            self.backbone_model = AutoModelForCausalLM.from_pretrained(config.backbone_model_name_or_path)
            self.backbone_model = self.backbone_model.half() if kwargs.get('fp16', False) else (self.backbone_model.bfloat16() if kwargs.get('bf16', False) else self.backbone_model)
        else:
            self.backbone_model = AutoModelForCausalLM.from_pretrained(config.backbone_model_name_or_path)
            self.backbone_model = self.backbone_model.half() if kwargs.get('fp16', False) else (self.backbone_model.bfloat16() if kwargs.get('bf16', False) else self.backbone_model)
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_uniform_(reward_head.weight, gain=1)
        torch.nn.init.zeros_(reward_head.bias)
        self.reward_head = reward_head.to(next(self.backbone_model.parameters()).device)

        # enn
        eta_net = Epinet(hid_rep_size = hidden_size, ref_size = self.ref_size, hidden_size = self.enn_hidden_size, output_size = self.enn_output_size, gain = self.enn_gain) 
        p_net = Epinet(hid_rep_size = hidden_size, ref_size = self.ref_size, hidden_size = self.enn_hidden_size, output_size = self.enn_output_size, gain = self.enn_gain, lmbda = self.lmbda)
        self.eta_net = eta_net.to(next(self.backbone_model.parameters()).device)
        self.p_net = p_net.to(next(self.backbone_model.parameters()).device)
        # detach the p_network
        for param in self.p_net.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, z_samples, return_full_z = False, return_dict=True, **kwargs):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        outputs = self.backbone_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        # print('input_ids', input_ids.shape)
        last_hidden_state = outputs.last_hidden_state
        last_hid_rep = last_hidden_state[:, -1, :]

        num_Z_size = z_samples.shape[0]
        model_out = self.reward_head(last_hid_rep)
        # print reward head weights and bias

        z_last_hid_rep = last_hid_rep.unsqueeze(1).repeat(1, num_Z_size, 1)
        eta = self.eta_net(z_last_hid_rep, z_samples, return_full_z).squeeze(-1)
        p = self.p_net(z_last_hid_rep, z_samples, return_full_z).squeeze(-1)
        if return_full_z:
            rewards_list = model_out + eta + p
            return ENNComponet(model_out = model_out, reward_list = rewards_list, eta_list = eta, p_list = p) if return_dict else (model_out, rewards_list, eta, p)
        else:
            eta = eta.view(-1, 1)
            p = p.view(-1, 1)
            rewards = model_out + eta + p
            rewards.view(-1, 1)

            return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)
    