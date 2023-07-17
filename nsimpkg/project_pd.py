import numpy as np

def is_pd(M):
    """
    Checks if matrix is in the positive definite cone using the Cholesky decomposition.

    Args:
    - M (np.matrix): matrix to check
    """

    try:
        _ = np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def project_pd(M, eps=1e-3, set_val=1e-3):
    """
    Projects a matrix to the positive definite cone. If the matrix is already in the positive definite cone, 
    it is returned unchanged.

    Args:
    - M (np.matrix): matrix to project
    - eps (float): tolerance to set negative and zero eigenvalues to
    - set (float): set negative and zero eigenvalues to this value


    Returns:
    - M (np.matrix): projected matrix
    """
    # project a matrix to the positive definite cone
    # M: a symmetric matrix
    # return: a symmetric matrix
    if is_pd(M):
        return M
    print("projecting")
    # projection to the positive definite cone
    eigval, eigvec = np.linalg.eigh(M)
    max_eigval = np.max(eigval)
    eigval[eigval <= eps] = set_val
    projected = eigvec.T @ np.diag(eigval) @ eigvec
    print(np.linalg.cond(projected))
    return projected