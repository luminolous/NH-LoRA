from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch.nn import functional as F


# Fast path for fixed shared-basis merge is only used when the shared basis is
# already very close to row-orthonormal. Above this tolerance we switch to a
# numerically safer least-squares solve instead of assuming A A^T == I exactly.
GRAM_FAST_PATH_TOLERANCE = 1e-4

# Row-basis rank detection / orthogonal-complement generation uses a small
# numerical epsilon so we do not pretend nearly dependent rows provide extra
# orthogonal capacity.
ORTHOGONALITY_EPS = 1e-6


@dataclass
class OrthogonalizationResult:
    rows: torch.Tensor
    requested_added_rank: int
    actual_added_rank: int
    orth_exhausted: bool
    orth_warning: str | None
    basis_rank: int
    available_rank: int


@dataclass
class FixedSharedMergeResult:
    solver: str
    gram_error: float
    reconstruction_error: float
    shared_a_preserved: bool


def gram_error(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix for Gram error, received shape={tuple(matrix.shape)}.")
    rows = int(matrix.size(0))
    if rows == 0:
        return matrix.new_zeros(())
    identity = torch.eye(rows, device=matrix.device, dtype=matrix.dtype)
    gram = matrix @ matrix.transpose(0, 1)
    return (gram - identity).abs().mean()


def row_orthonormal_matrix(
    num_rows: int,
    num_cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OrthogonalizationResult:
    requested = max(int(num_rows), 0)
    cols = max(int(num_cols), 0)
    actual = min(requested, cols)
    matrix = torch.zeros(requested, cols, device=device, dtype=dtype)
    warning = None
    if actual > 0:
        random_basis = torch.randn(cols, actual, device=device, dtype=dtype)
        q, _ = torch.linalg.qr(random_basis, mode="reduced")
        matrix[:actual] = q[:, :actual].transpose(0, 1)
    if actual < requested:
        warning = (
            f"Requested {requested} orthogonal rows but only {actual} fit inside the "
            f"input dimension {cols}."
        )
    return OrthogonalizationResult(
        rows=matrix,
        requested_added_rank=requested,
        actual_added_rank=actual,
        orth_exhausted=actual < requested,
        orth_warning=warning,
        basis_rank=0,
        available_rank=max(cols, 0),
    )


def row_basis_rank(rows: torch.Tensor, *, eps: float = ORTHOGONALITY_EPS) -> int:
    if rows.ndim != 2:
        raise ValueError(f"Expected rank-2 row basis, received shape={tuple(rows.shape)}.")
    if rows.numel() == 0 or rows.size(0) == 0:
        return 0
    q, r = torch.linalg.qr(rows.transpose(0, 1), mode="reduced")
    del q
    if r.numel() == 0:
        return 0
    diagonal = torch.diagonal(r, offset=0)
    return int((diagonal.abs() > float(eps)).sum().item())


def orthonormalize_rows(rows: torch.Tensor, *, eps: float = ORTHOGONALITY_EPS) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"Expected rank-2 row tensor, received shape={tuple(rows.shape)}.")
    if rows.numel() == 0 or rows.size(0) == 0:
        return rows.new_zeros((0, rows.size(1)))
    q, r = torch.linalg.qr(rows.transpose(0, 1), mode="reduced")
    diagonal = torch.diagonal(r, offset=0)
    rank = int((diagonal.abs() > float(eps)).sum().item())
    if rank <= 0:
        return rows.new_zeros((0, rows.size(1)))
    return q[:, :rank].transpose(0, 1)


def build_row_basis(
    row_groups: Iterable[torch.Tensor],
    *,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = ORTHOGONALITY_EPS,
) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    for group in row_groups:
        if group is None or group.numel() == 0:
            continue
        tensor = group.to(device=device, dtype=dtype)
        if tensor.ndim != 2 or tensor.size(1) != input_dim:
            raise ValueError(
                f"Expected row group with shape (*, {input_dim}), received {tuple(tensor.shape)}."
            )
        nonzero_mask = tensor.norm(dim=1) > float(eps)
        if bool(nonzero_mask.any()):
            collected.append(tensor[nonzero_mask])
    if not collected:
        return torch.zeros(0, input_dim, device=device, dtype=dtype)
    concatenated = torch.cat(collected, dim=0)
    return orthonormalize_rows(concatenated, eps=eps)


def orthogonal_rows_against_basis(
    *,
    requested_rows: int,
    input_dim: int,
    basis_rows: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = ORTHOGONALITY_EPS,
) -> OrthogonalizationResult:
    requested = max(int(requested_rows), 0)
    if requested <= 0:
        return OrthogonalizationResult(
            rows=torch.zeros(0, input_dim, device=device, dtype=dtype),
            requested_added_rank=0,
            actual_added_rank=0,
            orth_exhausted=False,
            orth_warning=None,
            basis_rank=0,
            available_rank=input_dim,
        )
    orth_basis = build_row_basis([basis_rows], input_dim=input_dim, device=device, dtype=dtype, eps=eps)
    basis_rank = int(orth_basis.size(0))
    available_rank = max(int(input_dim - basis_rank), 0)
    actual = min(requested, available_rank)
    warning = None
    if actual > 0:
        if basis_rank <= 0:
            result_rows = row_orthonormal_matrix(actual, input_dim, device=device, dtype=dtype).rows[:actual]
        else:
            q_complete, _ = torch.linalg.qr(orth_basis.transpose(0, 1), mode="complete")
            result_rows = q_complete[:, basis_rank : basis_rank + actual].transpose(0, 1).contiguous()
    else:
        result_rows = torch.zeros(0, input_dim, device=device, dtype=dtype)
    if actual < requested:
        warning = (
            f"Orthogonal complement exhausted: requested {requested} rows but only {actual} "
            f"fit after reserving basis rank {basis_rank} inside input dimension {input_dim}."
        )
    return OrthogonalizationResult(
        rows=result_rows,
        requested_added_rank=requested,
        actual_added_rank=actual,
        orth_exhausted=actual < requested,
        orth_warning=warning,
        basis_rank=basis_rank,
        available_rank=available_rank,
    )


def pairwise_overlap_stats(left_rows: torch.Tensor, right_rows: torch.Tensor) -> Dict[str, float]:
    if left_rows.numel() == 0 or right_rows.numel() == 0:
        return {"max_abs_cos": 0.0, "mean_sq_overlap": 0.0}
    normalized_left = F.normalize(left_rows, dim=1)
    normalized_right = F.normalize(right_rows, dim=1)
    overlap = normalized_left @ normalized_right.transpose(0, 1)
    return {
        "max_abs_cos": float(overlap.abs().max().item()),
        "mean_sq_overlap": float(overlap.pow(2).mean().item()),
    }


def fixed_basis_merge_update(
    shared_a: torch.Tensor,
    merged_update: torch.Tensor,
    *,
    gram_fast_path_tolerance: float = GRAM_FAST_PATH_TOLERANCE,
) -> tuple[torch.Tensor, FixedSharedMergeResult]:
    current_gram_error = float(gram_error(shared_a).item())
    if current_gram_error <= float(gram_fast_path_tolerance):
        shared_b = merged_update @ shared_a.transpose(0, 1)
        reconstruction = shared_b @ shared_a
        return shared_b, FixedSharedMergeResult(
            solver="fast_projection",
            gram_error=current_gram_error,
            reconstruction_error=float((reconstruction - merged_update).norm().item()),
            shared_a_preserved=True,
        )

    solution = torch.linalg.lstsq(shared_a.transpose(0, 1), merged_update.transpose(0, 1)).solution
    shared_b = solution.transpose(0, 1)
    reconstruction = shared_b @ shared_a
    return shared_b, FixedSharedMergeResult(
        solver="lstsq",
        gram_error=current_gram_error,
        reconstruction_error=float((reconstruction - merged_update).norm().item()),
        shared_a_preserved=True,
    )
