from cartridges.datasets import CartridgePerplexityDataset

class LaTeXPerplexityDataset(CartridgePerplexityDataset):

    class Config(CartridgePerplexityDataset.Config):
        _pass_as_config = True
            
        packing_mode: Literal["truncate", "pad"]="pad"
        packed_seq_length: int = 2048   
        user_prompt_prefix: list[str] | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer    
    
    
    def _get_element(self, elem_idx: int) -> CartridgeDatasetElement:

    
    