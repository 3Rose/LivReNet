"""Assembly of the linear tetrahedral elasticity operator."""

import numpy as np
from scipy import sparse


def _elasticity_matrix(young: float, poisson: float) -> np.ndarray:
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    matrix = np.zeros((6, 6), dtype=np.float64)
    matrix[:3, :3] = lam
    np.fill_diagonal(matrix[:3, :3], lam + 2.0 * mu)
    matrix[3:, 3:] = np.eye(3) * mu
    return matrix


def assemble_linear_tetra_stiffness(
    nodes: np.ndarray,
    tetrahedra: np.ndarray,
    *,
    young: float,
    poisson: float,
    chunk_size: int = 4096,
) -> sparse.csr_matrix:
    """Assemble a node-major 3-D P1 linear-elastic stiffness matrix."""

    nodes = np.asarray(nodes, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (T, 4)")
    if not 0.0 < float(poisson) < 0.5 or float(young) <= 0.0:
        raise ValueError("expected young > 0 and poisson in (0, .5)")

    constitutive = _elasticity_matrix(float(young), float(poisson))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    components = np.arange(3, dtype=np.int64)

    for start in range(0, len(tetrahedra), int(chunk_size)):
        cells = tetrahedra[start : start + int(chunk_size)]
        coordinates = nodes[cells]
        affine = np.concatenate(
            [np.ones((len(cells), 4, 1), dtype=np.float64), coordinates], axis=2
        )
        determinants = np.linalg.det(affine)
        valid = np.abs(determinants) > 1.0e-12
        if not np.all(valid):
            cells = cells[valid]
            affine = affine[valid]
            determinants = determinants[valid]
        if not len(cells):
            continue

        inverse = np.linalg.inv(affine)
        gradients = inverse[:, 1:, :].transpose(0, 2, 1)
        strain = np.zeros((len(cells), 6, 12), dtype=np.float64)
        for local_node in range(4):
            gx, gy, gz = (gradients[:, local_node, axis] for axis in range(3))
            base = 3 * local_node
            strain[:, 0, base] = gx
            strain[:, 1, base + 1] = gy
            strain[:, 2, base + 2] = gz
            strain[:, 3, base] = gy
            strain[:, 3, base + 1] = gx
            strain[:, 4, base + 1] = gz
            strain[:, 4, base + 2] = gy
            strain[:, 5, base] = gz
            strain[:, 5, base + 2] = gx
        volume = np.abs(determinants) / 6.0
        element = volume[:, None, None] * np.einsum(
            "tki,kl,tlj->tij", strain, constitutive, strain, optimize=True
        )
        dofs = (cells[:, :, None] * 3 + components[None, None, :]).reshape(
            len(cells), 12
        )
        rows.append(
            np.broadcast_to(dofs[:, :, None], element.shape)
            .reshape(-1)
            .astype(np.int32)
        )
        cols.append(
            np.broadcast_to(dofs[:, None, :], element.shape)
            .reshape(-1)
            .astype(np.int32)
        )
        values.append(element.reshape(-1))

    if not values:
        raise ValueError("tetrahedra contain no non-degenerate cells")
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * len(nodes), 3 * len(nodes)),
    ).tocsr()
    matrix.sum_duplicates()
    return ((matrix + matrix.T) * 0.5).tocsr()
