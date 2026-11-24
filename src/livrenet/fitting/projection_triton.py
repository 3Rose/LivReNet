"""Tiled exhaustive CUDA projection, including edges and degenerate triangles."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _dot(x, y, z, a, b, c):
    return x * a + y * b + z * c


@tr.jit
def _segment(px, py, pz, ax, ay, az, bx, by, bz):
    ex, ey, ez = bx - ax, by - ay, bz - az
    s = tl.minimum(
        1.0,
        tl.maximum(
            0.0,
            _dot(px - ax, py - ay, pz - az, ex, ey, ez)
            / tl.maximum(_dot(ex, ey, ez, ex, ey, ez), 1e-30),
        ),
    )
    dx, dy, dz = px - ax - s * ex, py - ay - s * ey, pz - az - s * ez
    return _dot(dx, dy, dz, dx, dy, dz), s


@tr.jit
def _project(
    V,
    F,
    P,
    D,
    FaceIds,
    U,
    W,
    N: tl.constexpr,
    NF: tl.constexpr,
    NP: tl.constexpr,
    NT: tl.constexpr,
    K: tl.constexpr,
):
    point, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    j = tile * K + tl.arange(0, K)
    valid = j < NF
    ia = tl.load(F + 3 * j, valid, other=0)
    ib = tl.load(F + 3 * j + 1, valid, other=0)
    ic = tl.load(F + 3 * j + 2, valid, other=0)
    ax = tl.load(V + batch * N * 3 + ia * 3)
    ay = tl.load(V + batch * N * 3 + ia * 3 + 1)
    az = tl.load(V + batch * N * 3 + ia * 3 + 2)
    bx = tl.load(V + batch * N * 3 + ib * 3)
    by = tl.load(V + batch * N * 3 + ib * 3 + 1)
    bz = tl.load(V + batch * N * 3 + ib * 3 + 2)
    cx = tl.load(V + batch * N * 3 + ic * 3)
    cy = tl.load(V + batch * N * 3 + ic * 3 + 1)
    cz = tl.load(V + batch * N * 3 + ic * 3 + 2)
    px = tl.load(P + point * 3)
    py = tl.load(P + point * 3 + 1)
    pz = tl.load(P + point * 3 + 2)
    # Edge candidates also cover triangles with zero area.
    d, s = _segment(px, py, pz, ax, ay, az, bx, by, bz)
    u, w = s, tl.full((K,), 0.0, tl.float32)
    dd, ss = _segment(px, py, pz, ax, ay, az, cx, cy, cz)
    take = dd < d
    u = tl.where(take, 0.0, u)
    w = tl.where(take, ss, w)
    d = tl.minimum(d, dd)
    dd, ss = _segment(px, py, pz, bx, by, bz, cx, cy, cz)
    take = dd < d
    u = tl.where(take, 1.0 - ss, u)
    w = tl.where(take, ss, w)
    d = tl.minimum(d, dd)
    ex, ey, ez = bx - ax, by - ay, bz - az
    fx, fy, fz = cx - ax, cy - ay, cz - az
    aa = _dot(ex, ey, ez, ex, ey, ez)
    ab = _dot(ex, ey, ez, fx, fy, fz)
    bb = _dot(fx, fy, fz, fx, fy, fz)
    ap = _dot(px - ax, py - ay, pz - az, ex, ey, ez)
    bp = _dot(px - ax, py - ay, pz - az, fx, fy, fz)
    # Cross-product denominator avoids subtracting nearly equal products.
    nx, ny, nz = ey * fz - ez * fy, ez * fx - ex * fz, ex * fy - ey * fx
    det = _dot(nx, ny, nz, nx, ny, nz)
    uu = (bb * ap - ab * bp) / tl.maximum(det, 1e-30)
    ww = (aa * bp - ab * ap) / tl.maximum(det, 1e-30)
    qx, qy, qz = (
        px - ax - uu * ex - ww * fx,
        py - ay - uu * ey - ww * fy,
        pz - az - uu * ez - ww * fz,
    )
    dd = _dot(qx, qy, qz, qx, qy, qz)
    take = (det > 1e-20) & (uu >= 0) & (ww >= 0) & (uu + ww <= 1) & (dd < d)
    u = tl.where(take, uu, u)
    w = tl.where(take, ww, w)
    d = tl.where(take, dd, d)
    d = tl.where(valid, d, float("inf"))
    slot = tl.argmin(d, 0, tie_break_left=True)
    out = (batch * NP + point) * NT + tile
    tl.store(D + out, tl.min(d, 0))
    tl.store(FaceIds + out, tile * K + slot)
    tl.store(U + out, tl.sum(tl.where(tl.arange(0, K) == slot, u, 0.0), 0))
    tl.store(W + out, tl.sum(tl.where(tl.arange(0, K) == slot, w, 0.0), 0))


def search(vertices, faces, points):
    b, n, _ = vertices.shape
    nf, np = len(faces), len(points)
    k = 256
    nt = tr.cdiv(nf, k)
    shape = (b, np, nt)
    d = torch.empty(shape, device=vertices.device, dtype=torch.float32)
    ids = torch.empty(shape, device=vertices.device, dtype=torch.long)
    u, w = torch.empty_like(d), torch.empty_like(d)
    _project[(np, nt, b)](
        vertices.float().contiguous(),
        faces.contiguous(),
        points.float().contiguous(),
        d,
        ids,
        u,
        w,
        n,
        nf,
        np,
        nt,
        k,
        enable_fp_fusion=False,
    )
    best = d.argmin(-1, keepdim=True)
    ids = ids.gather(-1, best).squeeze(-1)
    u = u.gather(-1, best).squeeze(-1)
    w = w.gather(-1, best).squeeze(-1)
    return ids, torch.stack((1 - u - w, u, w), -1)
