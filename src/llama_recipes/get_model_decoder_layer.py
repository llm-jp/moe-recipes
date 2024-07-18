from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer

from llama_recipes.models.deepseek_moe.modeling_deepseek import DeepseekDecoderLayer


def get_model_decoder_layer(
    model_name: str,
) -> type[MixtralDecoderLayer] | type[Qwen2MoeDecoderLayer] | type[DeepseekDecoderLayer]:
    if "Mixtral" in model_name:
        return MixtralDecoderLayer
    elif "Qwen" in model_name:
        return Qwen2MoeDecoderLayer
    elif "deepseek" in model_name:
        return DeepseekDecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
