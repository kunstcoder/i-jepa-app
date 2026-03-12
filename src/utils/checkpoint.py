import logging
from collections.abc import Mapping

import torch

logger = logging.getLogger(__name__)


def _strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str):
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.", "encoder.", "backbone.", "model."):
        state_dict = _strip_prefix_if_present(state_dict, prefix)
    return state_dict


def _pick_encoder_state(checkpoint: Mapping) -> Mapping:
    # official I-JEPA style
    if "encoder" in checkpoint:
        return checkpoint["encoder"]
    if "target_encoder" in checkpoint:
        logger.warning("'encoder' key not found, falling back to 'target_encoder'")
        return checkpoint["target_encoder"]

    # common wrappers
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"]

    return checkpoint


def load_ijepa_encoder(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load encoder weights from an I-JEPA checkpoint (or similarly packaged state dict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = _pick_encoder_state(checkpoint)

    if not isinstance(encoder_state, Mapping):
        raise TypeError(
            f"Unsupported checkpoint format at {checkpoint_path}: expected mapping, got {type(encoder_state)}"
        )

    encoder_state = _normalize_state_dict_keys(dict(encoder_state))
    msg = model.load_state_dict(encoder_state, strict=False)

    loaded_cnt = len(encoder_state) - len(msg.unexpected_keys)
    total_model = len(model.state_dict())
    if loaded_cnt == 0:
        raise RuntimeError(
            "No encoder weights were loaded. Check model_name/patch_size compatibility and checkpoint format."
        )

    if msg.missing_keys:
        logger.warning("Missing keys when loading encoder: %s", msg.missing_keys)
    if msg.unexpected_keys:
        logger.warning("Unexpected keys when loading encoder: %s", msg.unexpected_keys)

    logger.info(
        "Loaded I-JEPA encoder from %s (%d/%d tensors matched)",
        checkpoint_path,
        loaded_cnt,
        total_model,
    )
    return model
