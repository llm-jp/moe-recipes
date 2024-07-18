from transformers import (
    MixtralForCausalLM,
    Qwen2MoeForCausalLM,
    MixtralConfig,
    AutoModelForCausalLM,
)
import torch
from megatron_lm.megatron.global_vars import get_args


def get_model(
    model_name: str, use_cache: bool = False
) -> MixtralForCausalLM | Qwen2MoeForCausalLM | AutoModelForCausalLM:
    args = get_args()

    if "Mixtral" in model_name:
        model = MixtralForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            max_position_embeddings=args.seq_length,
            # ref: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json#L19
            output_router_logits=args.output_router_logits,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            use_cache=use_cache,
        )

    elif "Qwen" in model_name:
        model = Qwen2MoeForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            max_position_embeddings=args.seq_length,
            sliding_window=args.seq_length,
            # ref: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/config.json#L33
            output_router_logits=args.output_router_logits,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            use_cache=use_cache,
        )

    else:
        raise NotImplementedError("model not implemented")

    return model  # type: ignore
