from omegaconf import OmegaConf


def merge_configs_and_drop_duplicate_keys(pretrain_cfg, finetune_cfg):
    finetune_cfg_as_regular_dict = OmegaConf.to_container(
        finetune_cfg, resolve=True, throw_on_missing=True
    )
    model_cfg = finetune_cfg_as_regular_dict | pretrain_cfg

    error_keys = []
    for key in finetune_cfg:
        if key in pretrain_cfg and finetune_cfg[key] != pretrain_cfg[key]:
            error_keys.append(key)
    if len(error_keys) > 0:
        errors = {
            key: f"{pretrain_cfg[key]} != {finetune_cfg[key]}" for key in error_keys
        }
        raise ValueError(f"Pretrain and finetune config mismatches on {errors}")

    keys_to_drop = ["vocab_size", "pad_token_id", "cls_token_id", "sep_token_id"]
    for key in keys_to_drop:
        model_cfg.pop(key)
    return model_cfg
