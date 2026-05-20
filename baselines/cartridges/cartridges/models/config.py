import torch
from typing import Optional, Dict, List, Literal, Any, Type, Union
from transformers import PreTrainedModel
from pydrantic import BaseConfig
from pydantic import Field

class ModelConfig(BaseConfig):
    checkpoint_path: Optional[str] = None

    def instantiate(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def _load_checkpoint(
        self, 
        model: torch.nn.Module,
    ):
        if self.checkpoint_path is not None:
            # Load model
            # load the state dict, but remove the "model." prefix and all other keys from the
            # the PyTorch Lightning module that are not in the actual model
            print(f"Loading weights from {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path)

            model.load_state_dict({
                k[len("model."):]: v 
                for k, v in ckpt["state_dict"].items() 
                if k.startswith("model.")
            })




class PeftConfig(BaseConfig):
    """Configuration for Parameter-Efficient Fine-Tuning (PEFT) methods."""
    enabled: bool = False
    method: Literal['lora', 'prefix_tuning', 'prompt_tuning', 'p_tuning', 'adapter'] = 'lora'
    
    # LoRA-specific parameters
    r: int = 8  # rank of the update matrices
    alpha: int = 16  # scaling factor
    dropout: float = 0.0
    bias: Literal['none', 'all', 'lora_only'] = 'none'
    task_type: Literal['CAUSAL_LM', 'SEQ_CLS', 'SEQ_2_SEQ_LM'] = 'CAUSAL_LM'
    
    # Prefix Tuning parameters
    num_virtual_tokens: int = 20  # number of virtual tokens for prefix tuning
    encoder_hidden_size: Optional[int] = None  # hidden size for the encoder
    prefix_projection: bool = False  # whether to project the prefix
    
    # Prompt Tuning parameters
    prompt_tuning_init: Optional[Literal['TEXT', 'RANDOM']] = None
    prompt_tuning_init_text: Optional[str] = None
    
    # P-Tuning parameters
    encoder_reparameterization_type: Literal['MLP', 'LSTM'] = 'MLP'
    encoder_dropout: float = 0.0
    
    # Adapter parameters
    adapter_reduction_factor: int = 16
    adapter_non_linearity: Literal['relu', 'gelu', 'silu'] = 'relu'
    
    # Target modules/layers
    target_modules: Optional[List[str]] = None
    
    # Dictionary for any additional method-specific parameters
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    def get_peft_config(self):
        """Returns the appropriate PEFT config based on the specified method."""
        if not self.enabled:
            return None
            
        from peft import LoraConfig, PrefixTuningConfig, PromptTuningConfig, PromptEncoderConfig
        
        if self.method == 'lora':
            return LoraConfig(
                r=self.r,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias=self.bias,
                task_type=self.task_type,
                **self.extra_params
            )
        elif self.method == 'prefix_tuning':
            return PrefixTuningConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.prefix_projection,
                task_type=self.task_type,
                **self.extra_params
            )
        elif self.method == 'prompt_tuning':
            return PromptTuningConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                prompt_tuning_init=self.prompt_tuning_init,
                prompt_tuning_init_text=self.prompt_tuning_init_text,
                task_type=self.task_type,
                **self.extra_params
            )
        elif self.method == 'p_tuning':
            return PromptEncoderConfig(
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_reparameterization_type=self.encoder_reparameterization_type,
                encoder_hidden_size=self.encoder_hidden_size,
                encoder_dropout=self.encoder_dropout,
                task_type=self.task_type,
                **self.extra_params
            )
        
        return None


class HFModelConfig(ModelConfig):
    pretrained_model_name_or_path: Optional[str] = None
    load_kwargs: Optional[Dict] = Field(default_factory=dict)
    
    # PEFT configuration
    peft: PeftConfig = Field(default_factory=PeftConfig)

    # Whether to use the custom TrainableCache (prefix-tuning) or PEFT
    tuning_method: Literal['custom_prefix', 'peft'] = 'custom_prefix'

    model_cls: Optional[Type[PreTrainedModel]] = None
    attn_implementation: Optional[Literal['einsum', 'sdpa']] = None

    def instantiate(self):
        if self.model_cls is None:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
        else:
            model_cls = self.model_cls

        model = model_cls.from_pretrained(
            self.pretrained_model_name_or_path,
            **self.load_kwargs
        )

        if self.tuning_method == 'peft' and self.peft.enabled:
            from peft import get_peft_model
            peft_config = self.peft.get_peft_config()
            if peft_config is not None:
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
        
        return model
    