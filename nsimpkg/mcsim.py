import numpy as np

def mcsim(phi, d, N=1000):
    np.random.seed(0)
    s = d.sample(N)
    return np.mean(phi(s))

def log_rho(p, q, Nsamples = 1000):
    samples = q.sample(Nsamples)
    log_p = np.log(p.pdf(samples))
    log_q = np.log(q.pdf(samples))
    full_arr = 2 * (log_p - log_q)
    return np.mean(full_arr)

def rho(p, q, Nsamples = 1000):
    samples = q.sample(Nsamples)
    return np.mean((p.pdf(samples) / q.pdf(samples))**2)