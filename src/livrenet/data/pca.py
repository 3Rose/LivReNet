"""Exact zero-origin top-rank SVD via the smaller Gram eigenproblem."""

import numpy as np


def zero_origin_svd(fields, ranks):
    matrix = np.stack(fields).reshape(len(fields), -1).astype(np.float64)
    gram = matrix @ matrix.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][: max(ranks)]
    values = np.maximum(eigenvalues[order], 0.0)
    if len(values) < max(ranks) or np.any(values <= max(values[0], 1.0) * 1e-12):
        raise ValueError("Fit FEM span is too small for the requested PCA rank")
    components = (vectors[:, order].T @ matrix) / np.sqrt(values)[:, None]
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    result = {}
    for rank in ranks:
        basis = components[:rank].astype(np.float32)
        if not np.allclose(basis @ basis.T, np.eye(rank), atol=1e-4):
            raise ValueError("PCA numerical orthogonality check failed")
        result[rank] = {
            "components_flat": basis,
            "coefficient_scale": np.sqrt(values[:rank] / len(fields)).astype(
                np.float32
            ),
            "explained_energy_ratio": (
                values[:rank] / max(float(np.trace(gram)), 1e-12)
            ).astype(np.float32),
            "effective_rank": np.asarray(rank, np.int32),
        }
    return result
