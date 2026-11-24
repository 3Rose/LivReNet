"""One objective in millimetres for both networks and checkpoint selection."""

import torch


def components(prediction, target, identity, config):
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 3
    ):
        raise ValueError("Expected matching B x N x 3 displacement fields")
    if identity.shape != prediction.shape[:1]:
        raise ValueError("Expected one identity flag per field")
    magnitude = torch.linalg.vector_norm(target, dim=-1)
    maximum = magnitude.amax(-1)
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    peak_mask = (magnitude >= config["peak_fraction"] * maximum[:, None]) & (
        maximum[:, None] >= config["peak_minimum_mm"]
    )
    global_error = error.mean(-1)
    peak_error = (error * peak_mask).sum(-1) / peak_mask.sum(-1).clamp_min(1)
    k = max(1, int(prediction.shape[1] * config["identity_tail_fraction"]))
    # Penalize TOTAL predicted movement, not the residual correction alone.
    tail = torch.linalg.vector_norm(prediction, dim=-1).topk(k, dim=-1).values.mean(-1)
    identity_tail = tail * identity.to(prediction.dtype)
    total = (
        config["global_weight"] * global_error
        + config["peak_weight"] * peak_error
        + config["identity_tail_weight"] * identity_tail
    )
    return dict(
        global_error=global_error,
        peak_error=peak_error,
        identity_tail=identity_tail,
        total=total,
    )
