"""Two local PC blocks, two control blocks, exactly one 32x32 solve."""

from __future__ import annotations
import torch
from torch import nn

DEFAULT_CONFIG = {
    "width": 48,
    "pc_blocks": 2,
    "control_blocks": 2,
    "association_sigma_normalized": 0.15,
    "support_mass_scale": 8.0,
    "evidence_correction_cap_mm": 10.0,
    "signal_scale_mm": 2.5,
    "solve_ridge": 0.001,
    "solve_jitter": 0.00001,
}


def mlp(a, h, b):
    return nn.Sequential(nn.Linear(a, h), nn.SiLU(), nn.Linear(h, b))


class PointBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = mlp(2 * width + 4, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, xyz, knn):
        batch = torch.arange(len(h), device=h.device)[:, None, None]
        neighbor = h[batch, knn]
        center = h[:, :, None].expand_as(neighbor)
        rel = xyz[batch, knn] - xyz[:, :, None]
        message = self.message(
            torch.cat(
                (center, neighbor - center, rel, rel.norm(dim=-1, keepdim=True)), -1
            )
        )
        return self.norm(h + message.amax(2))


class ControlBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.message = mlp(2 * width + 4, width, width)
        self.update = mlp(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, graph):
        src, dst = graph["edge"]
        geom = graph["edge_geometry"].expand(len(h), -1, -1)
        msg = self.message(torch.cat((h[:, src], h[:, dst], geom), -1))
        aggregate = h.new_zeros(h.shape).index_add_(1, dst, msg) / graph["degree"]
        return self.norm(h + self.update(torch.cat((h, aggregate), -1)))


class CompactEvidencePCA(nn.Module):
    inference_inputs = frozenset(
        ("points", "knn", "candidates", "surface_residual_mm", "distance_mm")
    )

    def __init__(self, config=None):
        super().__init__()
        cfg = DEFAULT_CONFIG if config is None else config
        self.cfg = cfg
        w = cfg["width"]
        self.pc_encoder = mlp(7, w, w)
        self.pc_blocks = nn.ModuleList([PointBlock(w) for _ in range(cfg["pc_blocks"])])
        self.control_encoder = mlp(3, w, w)
        self.pair = mlp(2 * w + 8, w, w)
        self.match = nn.Linear(w, 1)
        self.evidence_encoder = mlp(2 * w + 4, w, w)
        self.control_blocks = nn.ModuleList(
            [ControlBlock(w) for _ in range(cfg["control_blocks"])]
        )
        self.correction = nn.Linear(w, 4)
        for head in (self.match, self.correction):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, inputs, graph, pc_messages=True, control_messages=True):
        if set(inputs) != self.inference_inputs:
            raise ValueError(
                "Only measured-PC geometry is accepted; labels are forbidden"
            )
        xyz = inputs["points"]
        residual = inputs["surface_residual_mm"]
        distance = inputs["distance_mm"]
        h = self.pc_encoder(torch.cat((xyz, residual / 10.0, distance / 10.0), -1))
        if pc_messages:
            for block in self.pc_blocks:
                h = block(h, xyz, inputs["knn"])
        ctrl = graph["control_xyz"]
        hc = self.control_encoder(ctrl)[None].expand(len(h), -1, -1)
        idx = inputs["candidates"]
        batch = torch.arange(len(h), device=h.device)[:, None, None]
        rel = xyz[:, :, None] - ctrl[idx]
        d = rel.norm(dim=-1, keepdim=True)
        hp = h[:, :, None].expand(-1, -1, idx.shape[-1], -1)
        pair = self.pair(
            torch.cat(
                (
                    hp,
                    hc[batch, idx],
                    rel,
                    d,
                    (residual / 10.0)[:, :, None].expand(-1, -1, idx.shape[-1], -1),
                    (distance / 10.0)[:, :, None].expand(-1, -1, idx.shape[-1], -1),
                ),
                -1,
            )
        )
        logits = -0.5 * (
            d.squeeze(-1) / self.cfg["association_sigma_normalized"]
        ).square() + self.match(pair).squeeze(-1)
        weights = logits.softmax(-1)
        n = len(ctrl)
        flat = idx.reshape(len(h), -1)
        weight = weights.reshape(len(h), -1, 1)

        def scatter(value):
            out = value.new_zeros(len(h), n, value.shape[-1])
            return out.scatter_add(
                1, flat[..., None].expand(-1, -1, value.shape[-1]), value * weight
            )

        mass = scatter(torch.ones_like(weight))
        pooled = scatter(hp.reshape(len(h), -1, h.shape[-1])) / mass.clamp_min(1e-8)
        point_evidence = (
            residual[:, :, None]
            .expand(-1, -1, idx.shape[-1], -1)
            .reshape(len(h), -1, 3)
        )
        evidence = scatter(point_evidence) / mass.clamp_min(1e-8)
        support = -torch.expm1(-mass / self.cfg["support_mass_scale"])
        hidden = self.evidence_encoder(
            torch.cat((hc, pooled, evidence / 10.0, support), -1)
        )
        if control_messages:
            for block in self.control_blocks:
                hidden = block(hidden, graph)
        adjustment = self.correction(hidden)
        signal = evidence.norm(dim=-1).topk(max(1, n // 10), dim=1).values.mean(1)
        signal_gate = torch.tanh(signal / self.cfg["signal_scale_mm"])[:, None, None]
        correction = (
            self.cfg["evidence_correction_cap_mm"]
            * torch.tanh(adjustment[..., :3])
            * signal_gate
        )
        evidence = evidence + correction
        support = (support * (1 + 0.5 * torch.tanh(adjustment[..., 3:]))).clamp(0, 1)
        # Shared observable numerical no-signal exit, not a GT identity detector.
        no_signal = distance.amax(dim=(1, 2)) <= 1e-4
        support = support * (~no_signal)[:, None, None]
        design = graph["control_basis"].double()  # [C*3, R], millimetres
        w = support.expand(-1, -1, 3).reshape(len(h), -1).double()
        e = evidence.reshape(len(h), -1).double()
        lhs = torch.einsum("nr,bn,ns->brs", design, w, design)
        reg = self.cfg["solve_ridge"] * graph["energy"].double() + self.cfg[
            "solve_jitter"
        ] * torch.eye(design.shape[1], device=h.device, dtype=torch.float64)
        rhs = torch.einsum("nr,bn,bn->br", design, w, e)
        coefficients = (
            torch.linalg.solve(lhs + reg[None], rhs[..., None]).squeeze(-1).float()
        )
        field = torch.einsum("br,rnd->bnd", coefficients, graph["basis"])
        return dict(
            displacement_mm=field,
            coefficients=coefficients,
            evidence_mm=evidence,
            support=support.squeeze(-1),
            correction_mm=correction,
            association=weights,
            numeric_no_signal=no_signal,
        )
