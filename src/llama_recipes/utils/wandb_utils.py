import torch
import wandb
import os
import time
import math
from typing import Any


def log_model_info(model: torch.nn.Module) -> None:
    model_config: dict[str, Any] = {}
    model_config["activation_function"] = model.config.hidden_act
    model_config["hidden_size"] = model.config.hidden_size
    model_config["model_type"] = model.config.model_type
    model_config["max_position_embeddings"] = model.config.max_position_embeddings
    model_config["num_attention_heads"] = model.config.num_attention_heads
    model_config["num_hidden_layers"] = model.config.num_hidden_layers
    model_config["model_architecture"] = model.config.architectures[0]

    print(f"model info: {model}")
    print(f"model config: {model.config}")
    wandb.config.update(model_config)

    # distributed training info
    world_size = int(os.environ["WORLD_SIZE"])
    wandb.config.update({"world_size": world_size})


def log_wandb(
    real_batch_size: int,
    real_seq_len: int,
    model: torch.nn.Module,
    accumulation_loss: float,
    load_balancing_loss: float,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    gradient_accumulation_steps: int,
    world_size: int,
    iteration_start_time: float,
) -> None:
    wandb_stats: dict[str, Any] = {}

    # training info
    wandb_stats["training/loss"] = accumulation_loss
    wandb_stats["training/load_balancing_loss"] = load_balancing_loss
    wandb_stats["training/perplexity"] = math.exp(accumulation_loss)
    # utils info
    batch_size: int = real_batch_size
    sequence_length: int = real_seq_len

    wandb_stats["utils/batch_size"] = batch_size
    wandb_stats["utils/global_batch_size"] = batch_size * world_size * gradient_accumulation_steps
    wandb_stats["utils/seq_len"] = sequence_length
    wandb_stats["utils/gradient_accumulation_steps"] = gradient_accumulation_steps
    wandb_stats["utils/iteration"] = iteration

    # optimizer info
    wandb_stats["optimizer/lr"] = optimizer.param_groups[0]["lr"]

    optimizer_states_1: list[float] = [0.0] * 8
    optimizer_states_2: list[float] = [0.0] * 4

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            # optimizer state が空の場合は logging しない
            if not optimizer.state:
                continue
            if "exp_avg_sq" not in optimizer.state[param].keys():
                continue

            optimizer_states_1[0] += (torch.norm(optimizer.state[param]["exp_avg_sq"]).item()) ** 2  # type: ignore
            optimizer_states_1[1] += (
                torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt()).item()  # type: ignore
            ) ** 2
            optimizer_states_1[2] += (torch.norm(optimizer.state[param]["exp_avg"]).item()) ** 2  # type: ignore
            optimizer_states_1[3] += (torch.norm(param).item()) ** 2  # type: ignore
            optimizer_states_1[4] += torch.norm(optimizer.state[param]["exp_avg_sq"], p=1).item()  # type: ignore
            optimizer_states_1[5] += torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt(), p=1).item()  # type: ignore
            optimizer_states_1[6] += torch.norm(optimizer.state[param]["exp_avg"], p=1).item()  # type: ignore
            optimizer_states_1[7] += torch.norm(param, p=1).item()
            optimizer_states_2[0] = max(
                optimizer_states_2[0],  # type: ignore
                abs(optimizer.state[param]["exp_avg_sq"].max().item()),  # type: ignore
                abs(optimizer.state[param]["exp_avg_sq"].min().item()),  # type: ignore
            )
            optimizer_states_2[1] = max(
                optimizer_states_2[1],
                optimizer.state[param]["exp_avg_sq"].sqrt().abs_().max().item(),  # type: ignore
            )
            optimizer_states_2[2] = max(
                optimizer_states_2[2],  # type: ignore
                abs(optimizer.state[param]["exp_avg"].max().item()),  # type: ignore
                abs(optimizer.state[param]["exp_avg"].min().item()),  # type: ignore
            )
            optimizer_states_2[3] = max(
                optimizer_states_2[3],
                abs(param.max().item()),  # type: ignore
                abs(param.min().item()),  # type: ignore
            )
    if optimizer.state:  # optimizer stateがない場合はloggingしない
        # rank:0でしかoptimizer stateをloggingしないので world sizeで割る必要はない
        wandb_stats["optimizer/variance_l2"] = optimizer_states_1[0] ** 0.5
        wandb_stats["optimizer/variance_sqrt_l2"] = optimizer_states_1[1] ** 0.5
        wandb_stats["optimizer/momentum_l2"] = optimizer_states_1[2] ** 0.5
        wandb_stats["optimizer/weight_l2"] = optimizer_states_1[3] ** 0.5
        wandb_stats["optimizer/variance_l1"] = optimizer_states_1[4]
        wandb_stats["optimizer/variance_sqrt_l1"] = optimizer_states_1[5]
        wandb_stats["optimizer/momentum_l1"] = optimizer_states_1[6]
        wandb_stats["optimizer/weight_l1"] = optimizer_states_1[7]
        wandb_stats["optimizer/variance_abs_max"] = optimizer_states_2[0]
        wandb_stats["optimizer/variance_sqrt_abs_max"] = optimizer_states_2[1]
        wandb_stats["optimizer/momentum_abs_max"] = optimizer_states_2[2]
        wandb_stats["optimizer/weight_abs_max"] = optimizer_states_2[3]

    # stats
    iteration_elapsed_time = time.perf_counter() - iteration_start_time

    tokens_per_sec = batch_size * sequence_length * gradient_accumulation_steps / iteration_elapsed_time * world_size
    wandb_stats["stats/1_iteration_time"] = iteration_elapsed_time
    wandb_stats["stats/tokens_per_sec"] = tokens_per_sec
    wandb_stats["stats/tokens_per_sec_per_gpu"] = tokens_per_sec / world_size

    checkpoint_activations_factor = 3

    num_layers: int = model.config.num_hidden_layers
    hidden_size: int = model.config.hidden_size
    vocab_size: int = model.config.vocab_size
    activation_func: str = model.config.hidden_act
    intermediate_size: int = model.config.intermediate_size
    num_experts: int = 1
    if hasattr(model.config, "num_experts_per_tok"):
        num_experts: int = model.config.num_experts_per_tok

    activation_function_factor: int = 4  # GELU
    if activation_func == "silu":
        activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)

    batch_size = batch_size * gradient_accumulation_steps
    num_query_groups: int = model.config.num_attention_heads / model.config.num_key_value_heads

    # tflops calculation
    flops_per_iteration: float = checkpoint_activations_factor * ((
        (2 + (2 * 3) + activation_function_factor * (intermediate_size / hidden_size) * num_experts) * batch_size * sequence_length * num_layers * (hidden_size**2)
    ) + (
        ((  # Attention matrix & attention over values
            4 * batch_size * (sequence_length ** 2) * hidden_size
        ) / num_query_groups
        ) +  # noqa: W504
        # lm-head: logit layer
        2 * batch_size * sequence_length * hidden_size * vocab_size)
    )
    tflops: float = flops_per_iteration / (iteration_elapsed_time * (10**12))
    wandb_stats["stats/tflops"] = tflops

    wandb.log(wandb_stats, step=iteration)

    print("------------------------------------------------------------------")
    print(f"iteration: {iteration} , TFLOPS: {tflops}, Tokens per sec: {tokens_per_sec}, Loss: {accumulation_loss}, load balancing loss: {load_balancing_loss}")
    print(
        "------------------------------------------------------------------",
        flush=True,
    )
