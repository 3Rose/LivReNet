"""CUDA implementation of spatial constraint selection."""

import triton as tr
import triton.language as tl


@tr.jit
def _greedy(
    X,
    Valid,
    Order,
    Out,
    P: tl.constexpr,
    COUNT: tl.constexpr,
    SEP2: tl.constexpr,
    K: tl.constexpr,
):
    s = tl.arange(0, K)
    order = tl.load(Order + s, s < P, other=0)
    valid = tl.load(Valid + order, s < P, other=0) & (s < P)
    x = tl.load(X + order * 3)
    y = tl.load(X + order * 3 + 1)
    z = tl.load(X + order * 3 + 2)
    chosen = tl.full((K,), False, tl.int1)
    blocked = tl.full((K,), False, tl.int1)
    total_valid = tl.sum(valid.to(tl.int32), 0)
    if total_valid <= COUNT:
        chosen = valid
    else:
        n = 0
        j = 0
        while (j < P) & (n < COUNT):
            active = tl.sum((valid & ~blocked & (s == j)).to(tl.int32), 0) > 0
            if active:
                px = tl.sum(tl.where(s == j, x, 0.0), 0)
                py = tl.sum(tl.where(s == j, y, 0.0), 0)
                pz = tl.sum(tl.where(s == j, z, 0.0), 0)
                chosen |= s == j
                blocked |= (x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (
                    z - pz
                ) < SEP2
                n += 1
            j += 1
        # Same two-pass semantics: spatially separated first, then fill in
        # confidence order when fewer than COUNT satisfy the separation.
        available = valid & ~chosen
        remaining_rank = tl.cumsum(available.to(tl.int32), 0)
        chosen |= available & (remaining_rank <= COUNT - n)
    tl.store(Out + order, chosen, s < P)
