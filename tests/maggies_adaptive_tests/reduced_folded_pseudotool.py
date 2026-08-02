"""Two-sided reduced folded eigenproblems for weak finite-element pencils.

For ``K : X -> Y'`` and a weak shift/embedding matrix ``M``, this module
reduces the weighted folded problem

    (K-zM)^* C^{-1} (K-zM) x = mu H x.

``H`` is the domain Gram (Riesz) matrix and ``C`` is the Riesz matrix whose
inverse defines the norm on residuals in ``Y'``.  For the natural problem
``H_0^1 -> H^{-1}``, use ``H=C=R_H1``.  For an L2-realised problem, use
``H=C=M``.

Given left/right bases U, V, the dense projection is

    S = U^* K V,  G = U^* M V,
    C_U = U^* C U, H_V = V^* H V,

and the reduced folded problem is

    (S-zG)^* C_U^{-1} (S-zG)c = mu H_V c.

This is exact for full bases.  For thin Krylov-Schur bases it is the natural
two-sided approximation.  No full folded matrix is assembled here.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReducedFoldedPencil:
    """Dense two-sided projection of a weighted weak pencil."""

    S: np.ndarray
    G: np.ndarray
    C_U: np.ndarray
    H_V: np.ndarray

    def __post_init__(self):
        shape = self.S.shape
        if self.S.ndim != 2 or shape[0] != shape[1]:
            raise ValueError("S must be square")
        if any(matrix.shape != shape for matrix in (self.G, self.C_U, self.H_V)):
            raise ValueError("all reduced matrices must have the same shape")

    def shifted_matrix(self, z):
        """Return ``U^*(K-zM)V = S-zG``."""
        return self.S - z * self.G

    def folded_matrix(self, z):
        """Return ``(S-zG)^* C_U^{-1} (S-zG)``."""
        T = self.shifted_matrix(z)
        return T.conj().T @ np.linalg.solve(self.C_U, T)

    def smallest_eigenvalue(self, z):
        """Smallest spectral value of the reduced weighted folded problem."""
        # Cholesky whitening converts the reduced generalised problem to an
        # ordinary SVD, retaining Hermitian positive-semidefinite structure.
        Lc = np.linalg.cholesky(self.C_U)
        Lh = np.linalg.cholesky(self.H_V)
        T = self.shifted_matrix(z)
        That = np.linalg.solve(Lc, T)
        That = np.linalg.solve(Lh.conj(), That.T).T
        return float(np.linalg.svd(That, compute_uv=False)[-1] ** 2)

    def smallest_singular_value(self, z):
        """Smallest weighted singular value of the reduced shifted pencil."""
        return self.smallest_eigenvalue(z) ** 0.5


def _gram_inner(left, gram, right):
    """Return ``left^* gram right`` for PETSc Vecs."""
    gram_right = right.duplicate()
    gram.mult(right, gram_right)
    return left.dot(gram_right)


def weighted_orthonormalize(vectors, gram, tol=1.0e-11):
    """Modified Gram--Schmidt with respect to a Hermitian SPD PETSc matrix."""
    basis = []
    for vector in vectors:
        q = vector.copy()
        for _ in range(2):
            for v in basis:
                q.axpy(-_gram_inner(v, gram, q), v)
        norm_sq = float(np.real(_gram_inner(q, gram, q)))
        if norm_sq > tol * tol:
            q.scale(norm_sq ** -0.5)
            basis.append(q)
    if not basis:
        raise ValueError("no linearly independent vectors remain after orthogonalisation")
    return basis


def project_two_sided(operator, shift, codomain_gram, domain_gram,
                      left_vectors, right_vectors, tol=1.0e-11):
    """Build the two-sided reduced folded pencil from PETSc matrices.

    Parameters
    ----------
    operator, shift
        PETSc matrices ``K`` and ``M`` defining ``K-zM``.
    codomain_gram, domain_gram
        PETSc Riesz matrices ``C`` and ``H`` in the full weighted problem.
    left_vectors, right_vectors
        Selected left/right Krylov-Schur or Ritz vectors.  They are separately
        orthonormalised in the C and H inner products before projection.

    Returns
    -------
    (reduced, U, V)
        ``reduced`` evaluates the small dense folded problem; ``U`` and ``V``
        are the corresponding normalised PETSc bases.
    """
    U = weighted_orthonormalize(left_vectors, codomain_gram, tol=tol)
    V = weighted_orthonormalize(right_vectors, domain_gram, tol=tol)
    if len(U) != len(V):
        raise ValueError("left and right reduced bases must have equal dimension")

    n = len(U)
    S = np.empty((n, n), dtype=complex)
    G = np.empty((n, n), dtype=complex)
    C_U = np.empty((n, n), dtype=complex)
    H_V = np.empty((n, n), dtype=complex)
    for j, vj in enumerate(V):
        Kv = vj.duplicate()
        Mv = vj.duplicate()
        operator.mult(vj, Kv)
        shift.mult(vj, Mv)
        for i, ui in enumerate(U):
            S[i, j] = ui.dot(Kv)
            G[i, j] = ui.dot(Mv)
            C_U[i, j] = _gram_inner(U[i], codomain_gram, U[j])
            H_V[i, j] = _gram_inner(V[i], domain_gram, V[j])

    # Preserve the Hermitian SPD structure despite distributed roundoff.
    C_U = 0.5 * (C_U + C_U.conj().T)
    H_V = 0.5 * (H_V + H_V.conj().T)
    return ReducedFoldedPencil(S, G, C_U, H_V), U, V
