"""Device-resident total-PCA objective, two starts and geometric safeguards.

This ports the final reference stage. Running it directly on the neural proposal
is a *different full fitter*, explicitly evaluated against the four-stage lite.
Fixed iteration schedules replace host-side early exits; inactive starts are
masked. No .cpu(), .numpy() or .item() occurs in the CUDA numerical loop.
"""

import torch
from .projection import closest

INPUT_KEYS = frozenset(
    (
        "template",
        "faces",
        "basis",
        "metric",
        "points",
        "initial",
        "fixed",
        "center",
        "mass",
        "coefficient_ridge_center",
    )
)


def tensors(packet, device="cuda", dtype=torch.float64):
    if set(packet) != INPUT_KEYS:
        raise ValueError("Only explicit inference inputs are accepted")
    return {
        k: torch.as_tensor(
            v, device=device, dtype=torch.long if k == "faces" else dtype
        )
        for k, v in packet.items()
    }


def solve(a, b):
    # info and finiteness are checked by device-side acceptance/fallback masks.
    return torch.linalg.solve_ex(a, b, check_errors=False).result


def safe(template, initial, fields, faces):
    first = (template + initial)[faces]
    second = (template[None] + fields)[:, faces]
    n0 = torch.linalg.cross(first[:, 1] - first[:, 0], first[:, 2] - first[:, 0])
    n1 = torch.linalg.cross(
        second[:, :, 1] - second[:, :, 0], second[:, :, 2] - second[:, :, 0]
    )
    ratio = n1.norm(dim=-1) / n0.norm(dim=-1).clamp_min(1e-12)
    return (
        torch.isfinite(fields).all(dim=(-2, -1))
        & (ratio.amin(-1) >= 0.1)
        & (((n1 * n0).sum(-1) < 0).to(fields.dtype).mean(-1) <= 0.001)
    )


@torch.inference_mode()
def refine(packet, settings, steps=6):
    if set(packet) != INPUT_KEYS:
        raise ValueError("Only explicit inference inputs are accepted")
    t, faces, b = (packet[k] for k in ("template", "faces", "basis"))
    original, fixed0, points = (packet[k] for k in ("initial", "fixed", "points"))
    rank = len(b)
    eye = torch.eye(rank, device=b.device, dtype=b.dtype)
    gram = torch.einsum("rnd,snd->rs", b, b)
    parallel = solve(gram + 1e-12 * eye, torch.einsum("rnd,nd->r", b, fixed0))
    orthogonal = fixed0 - torch.einsum("r,rnd->nd", parallel, b)
    initial = original - orthogonal
    center = packet["center"] + parallel
    init = solve(gram + 1e-12 * eye, torch.einsum("rnd,nd->r", b, initial))
    weight = packet["mass"] / len(points)
    ids, bary, _, normals = closest(t + initial, faces, points)
    design = (b[:, faces[ids[0]]] * bary[0][None, :, :, None]).sum(2).permute(1, 2, 0)
    jacobian = torch.einsum("ndr,nd->nr", design, normals[0])
    g = jacobian.T @ (weight * jacobian)
    eigen, q = torch.linalg.eigh(0.5 * (g + g.T))
    eigen = eigen.clamp_min(0)
    prior_value = float(settings["global_prior_ridge"])
    prior = (q * (prior_value * (1 - 0.99 * eigen / (eigen + prior_value)))) @ q.T
    metric = 0.5 * (packet["metric"] + packet["metric"].T)
    diagonal = metric.diagonal()
    # NumPy's interpolated median, without dynamic boolean indexing or a
    # quantile range check that reads a device scalar back on the host.
    positive = diagonal > 1e-10
    ordered = torch.where(positive, diagonal, torch.inf).sort().values
    count = positive.long().sum()
    lo = ((count - 1).clamp_min(0) // 2).reshape(1)
    hi = (count // 2).reshape(1)
    denominator = 0.5 * (ordered.gather(0, lo) + ordered.gather(0, hi)).squeeze(0)
    denominator = torch.where(count > 0, denominator, 1.0).clamp_min(1e-6)
    metric = metric / denominator
    coefficient_lambda = float(settings["coefficient_ridge"])
    ridge = float(settings["fem_ridge"]) * metric + coefficient_lambda * eye
    regularizer = ridge + prior
    constant = coefficient_lambda * (parallel @ parallel)

    def field(c):
        return torch.einsum("br,rnd->bnd", c, b)

    def evaluate(c):
        f = field(c)
        assoc = closest(t[None] + f, faces, points)
        residual = points[None] - assoc[2]
        objective = (
            weight * residual.square().sum((-2, -1))
            + torch.einsum("br,rs,bs->b", c, ridge, c)
            - 2 * coefficient_lambda * (c * parallel).sum(-1)
            + constant
            + torch.einsum("br,rs,bs->b", c - center, prior, c - center)
        )
        return objective, f, assoc

    current = torch.stack((init, torch.zeros_like(init)))
    objective, fields, assoc = evaluate(current)
    best_obj = objective[0].clone()
    best_field = initial.clone()  # Reference's initial fallback convention.
    active = torch.ones(2, device=b.device, dtype=torch.bool)
    acceptable = safe(t, initial, fields, faces)
    proposed = torch.where(acceptable, objective, torch.inf)
    j = proposed.argmin().reshape(1)
    proposed_obj = proposed.index_select(0, j).squeeze(0)
    proposed_field = fields.index_select(0, j).squeeze(0)
    improve = proposed_obj < best_obj
    best_field = torch.where(improve, proposed_field, best_field)
    best_obj = torch.minimum(best_obj, proposed_obj)
    accepted_count = torch.zeros(2, device=b.device, dtype=torch.long)
    numerical_failure = torch.zeros((), device=b.device, dtype=torch.bool)
    fractions = torch.tensor([1.0, 0.5, 0.25], device=b.device, dtype=b.dtype)
    for _ in range(steps):
        ids, bary, _, _ = assoc
        vertices = faces[ids]
        source = (t[vertices] * bary[..., None]).sum(-2)
        design = (b[:, vertices] * bary[None, :, :, :, None]).sum(3).permute(1, 2, 3, 0)
        a = design.reshape(2, -1, rank)
        y = (points[None] - source).reshape(2, -1)
        rhs = (
            weight * (a.transpose(1, 2) @ y[..., None]).squeeze(-1)
            + prior @ center
            + coefficient_lambda * parallel
        )
        solved = solve(weight * (a.transpose(1, 2) @ a) + regularizer, rhs)
        numerical_failure |= ~torch.isfinite(solved).all()
        # Keep invalid solves out of geometric projection kernels; report and
        # reject the case rather than letting an implementation failure hide
        # behind a seemingly valid geometry fallback.
        solved = torch.where(torch.isfinite(solved).all(-1)[:, None], solved, current)
        delta = solved - current
        largest = field(delta).norm(dim=-1).amax(-1)
        gain = (4.0 / largest.clamp_min(1e-12)).clamp_max(1.0)
        candidates = (
            current[:, None]
            + gain[:, None, None] * fractions[None, :, None] * delta[:, None]
        )
        trial, ff, _ = evaluate(candidates.reshape(6, rank))
        valid = (
            safe(t, initial, ff, faces).reshape(2, 3)
            & (trial.reshape(2, 3) < objective[:, None] - 1e-8)
            & active[:, None]
            & torch.isfinite(solved).all(-1)[:, None]
        )
        take = valid.any(-1)
        slot = valid.long().argmax(-1)
        next_c = candidates[torch.arange(2, device=b.device), slot]
        current = torch.where(take[:, None], next_c, current)
        active &= take
        accepted_count += take.long()
        # Retain first accepted backtracking step, not best of all trials.
        objective, fields, assoc = evaluate(current)
        proposed = torch.where(safe(t, initial, fields, faces), objective, torch.inf)
        j = proposed.argmin().reshape(1)
        proposed_obj = proposed.index_select(0, j).squeeze(0)
        proposed_field = fields.index_select(0, j).squeeze(0)
        improve = proposed_obj < best_obj
        best_field = torch.where(improve, proposed_field, best_field)
        best_obj = torch.minimum(best_obj, proposed_obj)
    fallback = ~safe(t, original, best_field[None], faces)[0]
    output = torch.where(fallback, original, best_field)
    # Detect numerical identity from observed geometry, never dataset labels.
    _, _, anchors, _ = closest(t, faces, points)
    no_signal = (points - anchors[0]).norm(dim=-1).amax() <= 1e-4
    output = torch.where(no_signal, torch.zeros_like(output), output)
    finite = torch.isfinite(output).all()
    output = torch.where(finite & ~numerical_failure, output, original)
    return output.float(), dict(
        objective=best_obj,
        geometry_fallback=fallback,
        numerical_fallback=~finite | numerical_failure,
        numeric_no_signal=no_signal,
        accepted_steps=accepted_count,
        GT_in_fit=False,
    )
