"""Greedy selection of spatially separated fitting constraints."""

import torch


def selected_mask(positions, confidence, valid, count=300, separation=2.0):
    valid = valid & (confidence > 0)
    if len(valid) <= count:
        return valid
    # Stable ties are explicit. The historical CPU code's unstable tie ordering
    # is not portable across backends; this is measured, not called bit-exact.
    order = torch.argsort(confidence, descending=True, stable=True)
    if positions.device.type != "cuda":
        chosen = []
        candidates = order[valid[order]].tolist()
        if len(candidates) <= count:
            return valid
        for index in candidates:
            if (
                not chosen
                or (positions[chosen] - positions[index]).norm(dim=-1).min()
                >= separation
            ):
                chosen.append(index)
            if len(chosen) == count:
                break
        used = set(chosen)
        chosen += [j for j in candidates if j not in used][: count - len(chosen)]
        result = torch.zeros_like(valid)
        result[chosen] = True
        return result
    from .selection_triton import _greedy, tr

    result = torch.zeros_like(valid)
    _greedy[(1,)](
        positions.contiguous(),
        valid.contiguous(),
        order.contiguous(),
        result,
        len(valid),
        count,
        separation * separation,
        tr.next_power_of_2(len(valid)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result
