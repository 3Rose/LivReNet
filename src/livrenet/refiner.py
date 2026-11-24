"""Three-level mesh graph and unrestricted residual displacement."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


def pool(h, owner, n):
    out = h.new_zeros((len(h), n, h.shape[-1]))
    out.index_add_(1, owner, h)
    return out / torch.bincount(owner, minlength=n).clamp_min(1)[None, :, None]


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.message = mlp(2 * h + 8, h, h)
        self.node = mlp(2 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, h, e, geometry):
        s, d = e
        m = self.message(torch.cat([h[:, s], h[:, d], geometry], -1))
        agg = h.new_zeros(h.shape)
        agg.index_add_(1, d, m)
        agg = agg / torch.bincount(d, minlength=h.shape[1]).clamp_min(1)[None, :, None]
        return self.norm(h + self.node(torch.cat([h, agg], -1)))


class MeshGraph(nn.Module):
    def __init__(self):
        super().__init__()
        width = 48
        self.encoder = nn.Sequential(
            nn.Linear(20, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            nn.ModuleList(Block(width) for _ in range(2)) for _ in range(3)
        )
        self.head = nn.Linear(width, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode(self, x, a):
        features = x["local"]
        h = self.encoder(features)
        ref = a["ref"]
        cur = ref[None] + x["u0"]
        states = []
        for level in range(3):
            e = a[f"e{level}"]
            s, d = e
            dr = ref[s] - ref[d]
            dx = cur[:, s] - cur[:, d]
            geom = torch.cat(
                [
                    dr[None].expand(len(h), -1, -1) / 20,
                    dr.norm(dim=-1)[None, :, None].expand(len(h), -1, -1) / 20,
                    dx / 20,
                    dx.norm(dim=-1, keepdim=True) / 20,
                ],
                -1,
            )
            for block in self.blocks[level]:
                h = (
                    checkpoint(block, h, e, geom, use_reentrant=False)
                    if self.training and torch.is_grad_enabled()
                    else block(h, e, geom)
                )
            states.append(h)
            if level < 2:
                owner = a[f"owner{level}"]
                n = int(a[f"n{level + 1}"])
                h = pool(h, owner, n)
                ref = pool(ref[None], owner, n)[0]
                cur = pool(cur, owner, n)
        return states[0] + (states[1] + states[2][:, a["owner1"]])[:, a["owner0"]]


class FreeG(nn.Module):
    def __init__(self, normalization):
        super().__init__()
        self.core = MeshGraph()
        self.register_buffer(
            "mean", torch.tensor(normalization["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "scale", torch.tensor(normalization["scale"], dtype=torch.float32)
        )
        if (
            self.mean.shape != (20,)
            or self.scale.shape != (20,)
            or not (self.scale > 0).all()
        ):
            raise ValueError("Invalid training-only feature normalization")

    def forward(self, inputs, graph):
        if set(inputs) != {"raw", "u0", "no_signal"}:
            raise ValueError("Observable inputs only")
        local = ((inputs["raw"] - self.mean) / self.scale).clamp(-8, 8)
        state = self.core.encode(dict(local=local, u0=inputs["u0"]), graph)
        delta = self.core.head(state)
        # no_signal is accepted for cache compatibility, never used as a gate.
        return inputs["u0"] + delta, delta
