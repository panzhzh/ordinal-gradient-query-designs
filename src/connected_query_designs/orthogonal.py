"""Positive-diagonal QR sampling of Haar orthogonal matrices."""

from __future__ import annotations
import torch


def positive_diagonal_qr(matrix):
    """Return the unique positive-diagonal QR for nonsingular real matrices.

    When matrix has iid standard Gaussian entries, Q has Haar distribution.
    The sign adjustment is the classical QR sampling construction.
    """
    if matrix.shape[-1] != matrix.shape[-2] or matrix.is_complex():
        raise ValueError("real square matrices required")
    q, r = torch.linalg.qr(matrix)
    diagonal = r.diagonal(dim1=-2, dim2=-1)
    if not (diagonal != 0).all():
        raise ValueError("nonsingular QR required")
    signs = diagonal.sign()
    return (q * signs[..., None, :], signs[..., :, None] * r)
