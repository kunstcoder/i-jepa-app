import logging

import torch

logger = logging.getLogger(__name__)


def load_ijepa_encoder(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Load the encoder weights from an I-JEPA checkpoint.

    The I-JEPA checkpoint contains keys: encoder, predictor, target_encoder,
    opt, scaler, epoch. Only the 'encoder' weights are needed for downstream.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "encoder" in checkpoint:
        encoder_state = checkpoint["encoder"]
    elif "target_encoder" in checkpoint:
        logger.warning("'encoder' key not found, falling back to 'target_encoder'")
        encoder_state = checkpoint["target_encoder"]
    else:
        # Assume the checkpoint itself is a raw state dict
        encoder_state = checkpoint

    msg = model.load_state_dict(encoder_state, strict=False)
    if msg.missing_keys:
        logger.warning("Missing keys when loading encoder: %s", msg.missing_keys)
    if msg.unexpected_keys:
        logger.warning("Unexpected keys when loading encoder: %s", msg.unexpected_keys)

    logger.info("Loaded I-JEPA encoder from %s", checkpoint_path)
    return model
