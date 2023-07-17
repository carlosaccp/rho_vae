import numpy as np

def partial_lq_m(x, p):
    """
    Computes the partial derivative of the log-likelihood function with respect to the mean.

    Args:
    - x (np.array): point at which to evaluate the partial derivative
    - p (NormalRV): distribution

    Returns:
    - partial (np.array): partial derivative of the log-likelihood function with respect to the mean
    """

    partial = x - np.reshape(p.mu, (len(p.mu), 1))
    return partial

def partial_lq_S(x, p):
    """
    Computes the partial derivative of the log-likelihood function with respect to the covariance matrix.

    Args:
    - x (np.array): point at which to evaluate the partial derivative
    - p (NormalRV): distribution

    Returns:
    - res (np.array): partial derivative of the log-likelihood function with respect to the covariance matrix
    """

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    n = x.shape[1]
    d = len(p.mu)
    res = np.zeros((d,d,n))
    for i in range(n):
        xi = x[:,i]
        res[:,:,i] = -0.5*((np.outer(xi, xi) - np.outer(p.mu, p.mu) - p.Sigma))
    return res
