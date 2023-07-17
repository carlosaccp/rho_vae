import scipy as sp
from tqdm import tqdm
from nsimpkg.random_variables import BetaRV
import numpy as np

def partial_log_alpha(alpha, beta, x):
    return -sp.special.digamma(alpha) + sp.special.digamma(alpha + beta) + np.log(x)

def partial_log_beta(alpha, beta, x):
    return -sp.special.digamma(beta) + sp.special.digamma(alpha + beta) + np.log(1 - x)

def SG_OAIS_beta(phi, pi, q0, nsamples, niter, alpha=1e-3, fixed=False):
    """
    Implement the SG-OAIS algorithm.

    Args:
    - phi (function): test function to integrate against
    - pi (NormalRV): target distribution
    - q0 (NormalRV): initial distribution
    - nsamples (int): number of samples to draw at each iteration
    - niter (int): number of iterations
    - alpha (float): learning rate scaling factor, default 1e-3

    Returns:
    - results (list): list of the results of the integration at each iteration
    - distributions (list): list of the distributions at each iteration
    """

    results = []
    distributions = [q0]
    for i in tqdm(range(niter), leave=True, position=0):
        q_theta = distributions[-1]
        lr = alpha if fixed else alpha/np.sqrt(i+1)
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        #update q_theta
        new_alpha = np.abs(q_theta.alpha + lr*np.mean(w2 * partial_log_alpha(q_theta.alpha, q_theta.beta, q_samples), axis=0))
        new_beta = np.abs(q_theta.beta + lr*np.mean(w2 * partial_log_beta(q_theta.alpha, q_theta.beta, q_samples), axis=0))
        new_dist = BetaRV(new_alpha, new_beta)

        distributions.append(new_dist)

    return np.array(results), distributions

def Adam_OAIS_beta(phi, pi, q0, nsamples, niter, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implement the SG-OAIS algorithm.

    Args:
    - phi (function): test function to integrate against
    - pi (NormalRV): target distribution
    - q0 (NormalRV): initial distribution
    - nsamples (int): number of samples to draw at each iteration
    - niter (int): number of iterations
    - alpha (float): learning rate scaling factor, default 1e-3
    - beta1 (float): Adam parameter, default 0.9
    - beta2 (float): Adam parameter, default 0.999
    - epsilon (float): Adam parameter, default 1e-8

    Returns:
    - results (list): list of the results of the integration at each iteration
    - distributions (list): list of the distributions at each iteration
    """

    results = []
    distributions = [q0]
    m_alphas = [0]
    m_betas = [0]
    v_alphas = [0]
    v_betas = [0]
    for i in tqdm(range(niter), leave=True, position=0):
        q_theta = distributions[-1]
        lr = alpha
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        #update q_theta Adam
        m_alpha = beta1*m_alphas[-1] + (1-beta1)*np.mean(w2 * partial_log_alpha(q_theta.alpha, q_theta.beta, q_samples), axis=0)
        m_beta = beta1*m_betas[-1] + (1-beta1)*np.mean(w2 * partial_log_beta(q_theta.alpha, q_theta.beta, q_samples), axis=0)
        v_alpha = beta2*v_alphas[-1] + (1-beta2)*np.mean(w2 * partial_log_alpha(q_theta.alpha, q_theta.beta, q_samples)**2, axis=0)
        v_beta = beta2*v_betas[-1] + (1-beta2)*np.mean(w2 * partial_log_beta(q_theta.alpha, q_theta.beta, q_samples)**2, axis=0)
        m_alphas.append(m_alpha)
        m_betas.append(m_beta)
        v_alphas.append(v_alpha)
        v_betas.append(v_beta)
        m_hat_alpha = m_alpha/(1-beta1**(i+1))
        m_hat_beta = m_beta/(1-beta1**(i+1))
        v_hat_alpha = v_alpha/(1-beta2**(i+1))
        v_hat_beta = v_beta/(1-beta2**(i+1))
        new_alpha = np.abs(q_theta.alpha + lr*m_hat_alpha/(np.sqrt(v_hat_alpha) + epsilon))
        new_beta = np.abs(q_theta.beta + lr*m_hat_beta/(np.sqrt(v_hat_beta) + epsilon))
        new_dist = BetaRV(new_alpha, new_beta)
        distributions.append(new_dist)


    return np.array(results), distributions

def AdaGrad_OAIS_beta(phi, pi, q0, nsamples, niter, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implement the SG-OAIS algorithm.

    Args:
    - phi (function): test function to integrate against
    - pi (NormalRV): target distribution
    - q0 (NormalRV): initial distribution
    - nsamples (int): number of samples to draw at each iteration
    - niter (int): number of iterations
    - alpha (float): learning rate scaling factor, default 1e-3
    - beta1 (float): Adam parameter, default 0.9
    - beta2 (float): Adam parameter, default 0.999
    - epsilon (float): Adam parameter, default 1e-8

    Returns:
    - results (list): list of the results of the integration at each iteration
    - distributions (list): list of the distributions at each iteration
    """

    results = []
    distributions = [q0]
    # arrays for adagrad
    grad_alpha_sq = 0
    grad_beta_sq = 0
    for i in tqdm(range(niter), leave=True, position=0):
        q_theta = distributions[-1]
        lr = alpha
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        #update q_theta AdaGrad
        grad_alpha = np.mean(w2 * partial_log_alpha(q_theta.alpha, q_theta.beta, q_samples), axis=0)
        grad_beta = np.mean(w2 * partial_log_beta(q_theta.alpha, q_theta.beta, q_samples), axis=0)
        grad_alpha_sq += grad_alpha**2
        grad_beta_sq += grad_beta**2
        new_alpha = np.abs(q_theta.alpha + lr*grad_alpha/(np.sqrt(grad_alpha_sq) + epsilon))
        new_beta = np.abs(q_theta.beta + lr*grad_beta/(np.sqrt(grad_beta_sq) + epsilon))
        new_dist = BetaRV(new_alpha, new_beta)
        distributions.append(new_dist)
    return np.array(results), distributions