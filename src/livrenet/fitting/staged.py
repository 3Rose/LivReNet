"""GPU implementations of base/posterior/coherent and reference total-PCA.

Keep PCA/FEM equations, bounded candidate faces, trimming, spatial selection,
IRLS, coherent weights, data-mass computation and the total-stage objective.
Confidence ties use deterministic stable order. Fidelity is evaluated per stage;
this is not asserted to be a bit-exact port before the checks establish it.
"""

import torch
from .solver import refine as total, safe, solve
from .projection import closest
from .kernels import selected_mask


def positive_median(metric):
    diagonal = metric.diagonal()
    valid = diagonal > 1e-10
    ordered = torch.where(valid, diagonal, torch.inf).sort().values
    count = valid.long().sum()
    index = ((count - 1).clamp_min(0) // 2).reshape(1)
    return torch.where(count > 0, ordered.gather(0, index).squeeze(0), 1.0).clamp_min(
        1e-6
    )


def quantile_valid(value, valid, q):
    ordered = torch.where(valid, value, torch.inf).sort().values
    index = (valid.long().sum() - 1).clamp_min(0) * q
    lo = index.floor().long().reshape(1)
    hi = index.ceil().long().reshape(1)
    return (
        ordered.gather(0, lo) * (1 - index.frac())
        + ordered.gather(0, hi) * index.frac()
    ).squeeze(0)


def restricted_projection(points, triangles, valid):
    """Port of the original Ericson candidate projection in float64."""
    point, triangles = points.double()[:, None], triangles.double()
    a, b, c = triangles.unbind(-2)
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = point - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = point - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    assigned = torch.zeros_like(valid)
    bary = torch.zeros_like(triangles[..., 0, :])
    one, zero = torch.ones_like(d1), torch.zeros_like(d1)

    def place(mask, weights):
        nonlocal assigned, bary
        active = mask & ~assigned & valid
        bary = torch.where(active[..., None], weights, bary)
        assigned |= active

    place((d1 <= 0) & (d2 <= 0), torch.stack((one, zero, zero), -1))
    place((d3 >= 0) & (d4 <= d3), torch.stack((zero, one, zero), -1))
    v = d1 / (d1 - d3).clamp_min(1e-12)
    place((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack((1 - v, v, zero), -1))
    place((d6 >= 0) & (d5 <= d6), torch.stack((zero, zero, one), -1))
    w = d2 / (d2 - d6).clamp_min(1e-12)
    place((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack((1 - w, zero, w), -1))
    w = (d4 - d3) / (d4 - d3 + d5 - d6).clamp_min(1e-12)
    place(
        (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), torch.stack((zero, 1 - w, w), -1)
    )
    denominator = va + vb + vc
    denominator = torch.where(denominator.abs() > 1e-12, denominator, 1.0)
    v, w = vb / denominator, vc / denominator
    place(~assigned, torch.stack((1 - v - w, v, w), -1))
    anchors = (triangles * bary[..., None]).sum(-2)
    distance = (point - anchors).square().sum(-1)
    best = torch.where(valid, distance, torch.inf).argmin(-1)
    row = torch.arange(len(points), device=points.device)
    return anchors[row, best].float(), bary[row, best].float(), best


def constraints(obs, coefficient, fixed, settings, indices, trim=0.9):
    b, scale = obs["basis"], obs["scale"]
    field = torch.einsum("r,rnd->nd", coefficient, b) + fixed
    current = obs["template_norm"] + field / scale
    local_faces = obs["candidate_faces"][indices, : int(settings["candidate_faces"])]
    local_valid = obs["candidate_valid"][indices, : int(settings["candidate_faces"])]
    mesh_faces = obs["faces"][local_faces]
    raw = obs["raw"][indices]
    anchor, bary, slot = restricted_projection(raw, current[mesh_faces], local_valid)
    row = torch.arange(len(indices), device=indices.device)
    chosen_vertices = mesh_faces[row, slot]
    source = (obs["template_norm"][chosen_vertices] * bary[..., None]).sum(1)
    fixed_at = (fixed[chosen_vertices] * bary[..., None]).sum(1)
    target = (raw - source) * scale - fixed_at
    distance = (raw - anchor).norm(dim=-1) * scale
    valid = local_valid[row, slot] & torch.isfinite(distance)
    cutoff = quantile_valid(distance, valid, trim).clamp_min(
        float(settings["minimum_trim_mm"])
    )
    valid &= distance <= cutoff
    confidence = 1 / (
        1 + (distance / float(settings["correspondence_delta_mm"])).square()
    )
    confidence = torch.where(valid, confidence, 0.0)
    triangle = current[chosen_vertices]
    normal = torch.linalg.cross(
        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    )
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    design3 = (b[:, chosen_vertices] * bary[None, :, :, None]).sum(2).permute(1, 2, 0)
    return dict(
        design=torch.einsum("pdr,pd->pr", design3, normal),
        target=(target * normal).sum(-1),
        confidence=confidence,
        valid=valid,
        positions=source * scale,
        face_vertices=chosen_vertices,
        bary=bary,
        indices=indices,
    )


def weights_and_mass(con, settings):
    selected = selected_mask(
        con["positions"],
        con["confidence"],
        con["valid"],
        int(settings["constraint_count"]),
        float(settings["minimum_separation_mm"]),
    )
    weights = torch.where(selected, con["confidence"].clamp_min(1e-6), 0.0)
    weights = weights / weights.amax().clamp_min(1e-6)
    return weights, weights.sum()


def irls(
    con, prior_center, metric, settings, reference_mass=None, coherent_confidence=None
):
    if coherent_confidence is None:
        base, _ = weights_and_mass(con, settings)
    else:
        base = torch.where(con["valid"], coherent_confidence, 0.0)
        base = base * reference_mass / base.sum().clamp_min(1e-12)
    rank = len(prior_center)
    eye = torch.eye(rank, device=metric.device, dtype=metric.dtype)
    prior = float(settings.get("global_prior_ridge", 0.01))
    normalized_metric = 0.5 * (metric + metric.T)
    normalized_metric = normalized_metric / positive_median(normalized_metric)
    regularizer = (
        float(settings["fem_ridge"]) * normalized_metric
        + (float(settings["coefficient_ridge"]) + prior) * eye
    )
    weight = base
    design, target = con["design"], con["target"]
    coefficient = prior_center.clone()
    for _ in range(int(settings["irls_iterations"])):
        coefficient = solve(
            design.T @ (weight[:, None] * design) + regularizer,
            design.T @ (weight * target) + prior * prior_center,
        )
        residual = (design @ coefficient - target).abs().clamp_min(1e-6)
        weight = base * (float(settings["huber_delta_mm"]) / residual).clamp_max(1.0)
    return torch.where(base.sum() > 0, coefficient, prior_center)


def score(template, faces, points, field):
    _, _, anchor, _ = closest(template + field, faces, points)
    distance = (points - anchor[0]).norm(dim=-1)
    tail = distance.topk(max(1, (len(distance) + 9) // 10)).values.mean()
    return distance.mean() + 0.25 * torch.quantile(distance, 0.95) + 0.25 * tail


def coherence(points, residual):
    distance = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    dd, index = distance.topk(min(17, len(points)), largest=False)
    magnitude = residual.norm(dim=-1)
    direction = residual / magnitude[:, None].clamp_min(1e-8)
    valid = (index != torch.arange(len(points), device=points.device)[:, None]) & (
        dd <= 8.0
    )
    cosine = (direction[:, None] * direction[index]).sum(-1)
    supported = valid & (cosine >= 0.5) & (magnitude[index] >= 0.5 * magnitude[:, None])
    count = supported.sum(-1)
    fraction = count / valid.sum(-1).clamp_min(1)
    coherent = ((fraction - 0.4) / 0.4).clamp(0, 1) * (count >= 4)
    signal = ((magnitude - 1.5) / 1.5).clamp(0, 1)
    return 1 + 3 * coherent * signal


@torch.inference_mode()
def fit(packet, obs, settings, caps=(2, 1, 1, 6)):
    b, metric = obs["basis"], obs["metric"]
    c0 = packet["center"].float()
    coefficient, prior = c0.clone(), c0.clone()
    initial = packet["initial"].float()
    fixed = initial - torch.einsum("r,rnd->nd", c0, b)
    base_settings = dict(settings, global_prior_ridge=0.0)
    for _ in range(caps[0]):
        con = constraints(obs, coefficient, fixed, settings, obs["fit_indices"])
        solved = irls(con, prior, metric, base_settings)
        coefficient = coefficient + float(settings["damping"]) * (solved - coefficient)
    base_field = torch.einsum("r,rnd->nd", coefficient, b) + fixed
    prior = coefficient.clone()
    for _ in range(caps[1]):
        con = constraints(obs, coefficient, fixed, settings, obs["fit_indices"])
        coefficient = coefficient + float(settings["damping"]) * (
            irls(con, prior, metric, settings) - coefficient
        )
    posterior = torch.einsum("r,rnd->nd", coefficient, b) + fixed
    # Preserve historical double projection followed by float32 coherent solve.
    bd = b.double()
    eye = torch.eye(len(b), device=b.device, dtype=torch.float64)
    center = solve(
        torch.einsum("rnd,snd->rs", bd, bd) + 1e-12 * eye,
        torch.einsum("rnd,nd->r", bd, (posterior - fixed).double()),
    )
    coherent_center = center.float()
    coefficient = coherent_center.clone()
    con = constraints(obs, coefficient, fixed, settings, obs["fit_indices"])
    _, mass = weights_and_mass(con, settings)
    template = obs["template_norm"] * obs["scale"]
    points = obs["raw"] * obs["scale"]
    best, best_score = posterior, score(template, obs["faces"], points, posterior)
    for _ in range(caps[2]):
        con = constraints(
            obs, coefficient, fixed, settings, obs["all_indices"], trim=1.0
        )
        field = torch.einsum("r,rnd->nd", coefficient, b) + fixed
        vertices = con["face_vertices"]
        anchor = ((template + field)[vertices] * con["bary"][..., None]).sum(1)
        weight = coherence(points[con["indices"]], points[con["indices"]] - anchor)
        solved = irls(
            con,
            coherent_center,
            metric,
            settings,
            reference_mass=mass,
            coherent_confidence=weight,
        )
        delta = solved - coherent_center
        largest = torch.einsum("r,rnd->nd", delta, b).norm(dim=-1).amax()
        gain = (4.0 / largest.clamp_min(1e-8)).clamp_max(1.0)
        candidate_c = coherent_center + gain * delta
        candidate = torch.einsum("r,rnd->nd", candidate_c, b) + fixed
        valid = safe(template, posterior, candidate[None], obs["faces"])[0]
        candidate_score = score(template, obs["faces"], points, candidate)
        take = valid & (candidate_score < best_score - 1e-6)
        best = torch.where(take, candidate, best)
        best_score = torch.where(take, candidate_score, best_score)
        coefficient = torch.where(valid, candidate_c, coefficient)
    coherent = best
    # Recompute mass at coherent output exactly as the CPU pipeline does.
    final_c = solve(
        torch.einsum("rnd,snd->rs", bd, bd) + 1e-12 * eye,
        torch.einsum("rnd,nd->r", bd, (coherent - fixed).double()),
    ).float()
    con = constraints(obs, final_c, fixed, settings, obs["fit_indices"])
    _, mass = weights_and_mass(con, settings)
    final_packet = dict(
        packet,
        initial=coherent.double(),
        fixed=fixed.double(),
        center=center,
        mass=mass.double(),
        coefficient_ridge_center=torch.zeros_like(center),
    )
    output, diag = total(final_packet, settings, steps=caps[3])
    # Identity detection uses the existing observable input gate, never labels.
    output = torch.where(obs["numeric_no_signal"], torch.zeros_like(output), output)
    finite = torch.isfinite(output).all()
    output = torch.where(finite, output, initial)
    diag["numerical_fallback"] |= ~finite
    return output, dict(
        **diag,
        base_field=base_field,
        posterior_field=posterior,
        coherent_field=coherent,
        reference_mass=mass,
    )
