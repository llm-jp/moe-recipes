from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer


def get_model_decoder_layer(
    model_name: str,
) -> type[MixtralDecoderLayer] | type[Qwen2MoeDecoderLayer]:
    if "Mixtral" in model_name:
        return MixtralDecoderLayer
    elif "Qwen" in model_name:
        return Qwen2MoeDecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
