import os
import pdb
import fire
import random
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers.utils import logging

from model import ReconModel, DecoderLayer
from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

logger = logging.get_logger(__name__)


class Model_Config():
    def __int__(self):
        pass

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    dataset_config = generate_dataset_config(train_config, kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(world_size)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    config = Model_Config()
    config.hidden_size = 768
    config.intermediate_size = 3072
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    config.num_hidden_layers = 12
    config.img_size = dataset_config.img_size
    config.patch_size = dataset_config.patch_size
    config.norm_pixel_loss = train_config.norm_pixel_loss
    config.pos_type = train_config.pos_type

    model = ReconModel(config)

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)

        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        #model = model.to(rank)
        #model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)

    elif not train_config.enable_fsdp:
        model.to("cuda")

    # dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=False,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        train_config.gradient_accumulation_steps,
        train_config,
        dataset_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)