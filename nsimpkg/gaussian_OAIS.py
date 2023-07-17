import numpy as np
from tqdm import tqdm
from nsimpkg.project_pd import project_pd
from nsimpkg.proposal_updates import partial_lq_m, partial_lq_S
from nsimpkg.random_variables import NormalRV

def SG_OAIS(phi, pi, q0, nsamples, niter, alpha=1e-3, proj_eps=1e-6, proj_set=1e-3, fixed=False):
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
        lr = alpha/np.sqrt(i+1)
        if fixed:
            lr = alpha
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        #update q_theta
        partial_m = partial_lq_m(q_samples, q_theta)
        update_m = -np.mean(w2 * partial_m, axis=1)
        new_m = q_theta.m - lr*update_m

        partial_S = partial_lq_S(q_samples, q_theta)
        update_S = -np.mean(w2 * partial_S, axis=2)
        new_S = project_pd(q_theta.S - lr*update_S, eps=proj_eps, set_val=proj_set)

        new_dist = NormalRV(new_m, new_S, natural=True)
        distributions.append(new_dist)

    q_theta = distributions[-1]
    lr = alpha/np.sqrt(i+1) if fixed else alpha
    q_samples = q_theta.sample(nsamples)
    w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
    w2 = w**2
    phi_samples = phi(q_samples)
    integral = np.mean(w*phi_samples)
    results.append(integral)

    return np.array(results), distributions

def Adam_OAIS(phi, pi, q0, nsamples, niter, alpha=1e-3, beta1=0.9, beta2=0.999, proj_eps=1e-6, proj_set=1e-3):
    """
    Implement the ADAM-OAIS algorithm.

    Args:
    - phi (function): test function to integrate against
    - pi (NormalRV): target distribution
    - q0 (NormalRV): initial distribution
    - nsamples (int): number of samples to draw at each iteration
    - niter (int): number of iterations
    - alpha (float): learning rate, default 1e-3
    - beta1 (float): exponential decay rate for the first moment estimates, default 0.9
    - beta2 (float): exponential decay rate for the second moment estimates, default 0.999

    Returns:
    - results (list): list of the results of the integration at each iteration
    - distributions (list): list of the distributions at each iteration
    """

    results = []
    distributions = [q0]
    mus_m = [np.zeros(q0.mu.shape)]
    vs_m = [np.zeros(q0.mu.shape)]
    M_Sarr = [np.zeros(q0.Sigma.shape)]
    V_Sarr = [np.zeros(q0.Sigma.shape)]
    for i in tqdm(range(niter), leave=True, position=0):
        q_theta = distributions[-1]
        lr = alpha
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        partial_m = partial_lq_m(q_samples, q_theta)
        update_m = -np.mean(w2 * partial_m, axis=1)
        partial_S = partial_lq_S(q_samples, q_theta)
        update_S = -np.mean(w2 * partial_S, axis=2)

        mu_m = beta1 * mus_m[-1] + (1-beta1) * update_m
        mus_m.append(mu_m)
        v_m = beta2 * vs_m[-1] + (1-beta2) * (update_m)**2
        vs_m.append(v_m)

        mu_hat_m = mu_m / (1-beta1**(i+1))
        v_hat_m = v_m / (1-beta2**(i+1)) 
        new_m = q_theta.m - lr*(mu_hat_m/np.sqrt(v_hat_m[-1])+1e-8)
        
        M_t = beta1 * M_Sarr[-1] + (1-beta1) * update_S
        M_Sarr.append(M_t)
        V_t = beta2 * V_Sarr[-1] + (1-beta2) * (update_S)**2
        V_Sarr.append(V_t)

        M_hat = M_t / (1-beta1**(i+1))
        V_hat = V_t / (1-beta2**(i+1))
        new_S = project_pd(q_theta.S - lr*(M_hat/np.sqrt(V_hat)+1e-8), eps=proj_eps, set_val=proj_set)

        new_dist = NormalRV(new_m, new_S, natural=True)
        distributions.append(new_dist)

    q_theta = distributions[-1]
    lr = alpha
    q_samples = q_theta.sample(nsamples)
    w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
    w2 = w**2
    phi_samples = phi(q_samples)
    integral = np.mean(w*phi_samples)
    results.append(integral)

    return np.array(results), distributions

def AdaGrad_OAIS(phi, pi, q0, nsamples, niter, alpha=1e-3, beta1=0.9, beta2=0.999, proj_eps=1e-6, proj_set=1e-3):
    """
    Implement the ADAM-OAIS algorithm.

    Args:
    - phi (function): test function to integrate against
    - pi (NormalRV): target distribution
    - q0 (NormalRV): initial distribution
    - nsamples (int): number of samples to draw at each iteration
    - niter (int): number of iterations
    - alpha (float): learning rate, default 1e-3
    - beta1 (float): exponential decay rate for the first moment estimates, default 0.9
    - beta2 (float): exponential decay rate for the second moment estimates, default 0.999

    Returns:
    - results (list): list of the results of the integration at each iteration
    - distributions (list): list of the distributions at each iteration
    """

    results = []
    distributions = [q0]
    G_m = [np.zeros(q0.mu.shape)]
    G_Sarr = [np.zeros(q0.Sigma.shape)]
    for i in tqdm(range(niter), leave=True, position=0):
        q_theta = distributions[-1]
        lr = alpha
        q_samples = q_theta.sample(nsamples)
        w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
        w2 = w**2
        phi_samples = phi(q_samples)
        integral = np.mean(w*phi_samples)
        results.append(integral)


        partial_m = partial_lq_m(q_samples, q_theta)
        update_m = -np.mean(w2 * partial_m, axis=1)
        partial_S = partial_lq_S(q_samples, q_theta)
        update_S = -np.mean(w2 * partial_S, axis=2)

        G_m.append(G_m[-1] + update_m**2)
        G_Sarr.append(G_Sarr[-1] + update_S**2)

        new_m = q_theta.m - lr*(update_m/np.sqrt(G_m[-1])+1e-8)

        new_S = project_pd(q_theta.S - lr*(update_S/np.sqrt(G_Sarr[-1])+1e-8), eps=proj_eps, set_val=proj_set)

        new_dist = NormalRV(new_m, new_S, natural=True)
        distributions.append(new_dist)

    q_theta = distributions[-1]
    lr = alpha
    q_samples = q_theta.sample(nsamples)
    w = pi.pdf(q_samples) / q_theta.pdf(q_samples)
    w2 = w**2
    phi_samples = phi(q_samples)
    integral = np.mean(w*phi_samples)
    results.append(integral)


    return np.array(results), distributions