import numpy as np
import matplotlib.transforms as transforms
import scipy as sp
from matplotlib.patches import Ellipse

class NormalRV:
    """
    Class for a multivariate normal random variable.
    """

    def __init__(self, mu, Sigma, natural = False):
        """
        Constructor. Sets:
        - mu (np.array): mean
        - Sigma (np.matrix): covariance matrix
        - S (np.matrix): inverse of Sigma
        - m (np.array): S@mu
        """
        if natural:
            self.m = mu
            self.S = Sigma
            self.Sigma = np.linalg.inv(self.S)
            self.mu = self.Sigma @ self.m

        else:
            self.mu = mu
            self.Sigma = Sigma
            self.S = np.linalg.inv(Sigma)
            self.m = self.S @ self.mu
        self.dim = len(mu)
    
    def pdf(self, x):
        """
        Evaluates the pdf at x. x can be a vector or a matrix; in the latter case, 
        the pdf is evaluated column-wise.

        Args:
        - x (np.array): point(s) at which to evaluate the pdf

        Returns:
        - pdf (np.array): pdf evaluated at x
        """
        try:
            mu = self.mu
            Sigma = self.Sigma
            pdf = sp.stats.multivariate_normal.pdf(x.T, mu, Sigma)
            return pdf

        except ValueError:
            print(Sigma)

    def sample(self, n):
        """
        Draws n samples from the distribution.

        Args:
        - n (int): number of samples to draw

        Returns:
        - samples (np.array): samples drawn from the distribution
        """

        return np.random.multivariate_normal(self.mu, self.Sigma, n).T
    
    def construct_ellipse(self, n_std, ax, color, label, plot=True ):
        pearson = self.Sigma[0, 1]/np.sqrt(self.Sigma[0, 0] * self.Sigma[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor=color, label=label)
        scale_x = np.sqrt(self.Sigma[0, 0]) * n_std
        scale_y = np.sqrt(self.Sigma[1, 1]) * n_std
        mu_x = self.mu[0]
        mu_y = self.mu[1]

        transform = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mu_x, mu_y)
        
        ellipse.set_transform(transform + ax.transData)
        if plot:
            return ax.add_patch(ellipse)
        else:
            return ellipse

        
    
    def __str__(self):
        """
        Returns a string representation of the distribution.
        """

        return "N(" + str(self.mu) + ", " + str(self.Sigma) + ")"
    
    def __repr__(self):
        """
        Returns a string representation of the distribution.
        """

        return "N(" + str(self.mu) + ", " + str(self.Sigma) + ")"
    
    def __add__(self, other):
        """
        Returns the sum of two normal random variables.
        """

        mu = self.mu + other.mu
        Sigma = self.Sigma + other.Sigma
        return NormalRV(mu, Sigma)
    
    # define multiplication with scalar
    def __mul__(self, other):
        """
        Returns the product of a normal random variable with a scalar.
        """

        mu = other * self.mu
        Sigma = other * self.Sigma
        return NormalRV(mu, Sigma)
    
    def __rmul__(self, other):
        """
        Returns the product of a normal random variable with a scalar.
        """

        mu = other * self.mu
        Sigma = other * self.Sigma
        return NormalRV(mu, Sigma)


class MixNormalRV():

    def __init__(self, weights, mus, Sigmas):
        if len(weights) != len(mus) or len(weights) != len(Sigmas):
            raise ValueError("Weight, mean and covariance matrix arrays must have the same length.")
        
        if sum(weights) != 1:
            raise ValueError("Weights must sum to 1.")
        self.mus = [np.array(mu) for mu in mus]
        self.Sigmas = [np.array(Sigma) for Sigma in Sigmas]

        self.weights = weights
        self.dim = len(self.mus[0])
        self.distributions  = [NormalRV(mus[i], Sigmas[i]) for i in range(len(weights))]
        self.n_mix = len(self.weights)
        avg_mu = np.zeros(self.mus[0].shape)
        avg_Sigma = np.zeros(self.Sigmas[0].shape)
        for i in range(self.n_mix):
            avg_mu += self.weights[i] * self.mus[i]
            avg_Sigma += self.weights[i] * self.Sigmas[i]
        
        self.avg_dist = NormalRV(avg_mu, avg_Sigma)



    def pdf(self, x):
        ret = 0
        for i in range(self.n_mix):
            ret += self.weights[i] * self.distributions[i].pdf(x)
        return ret
    
    def sample(self, n):
        samples = np.zeros((self.dim, n))
        c_weights = np.cumsum(self.weights)
        for i in range(n):
            u_weights = np.random.uniform(0, 1, self.n_mix)
            idx = np.sum(c_weights < u_weights)
            samples[:, i] = self.distributions[idx].sample(1).reshape(-1)
        return samples
    
    def __str__(self):
        return "Mixed Gaussian distribution with means " + str([dist.mu for dist in self.distributions]) + " and covariances " + str([dist.Sigma for dist in self.distributions]) + "."
    
    def __repr__(self):
        return "Mixed Gaussian distribution with means " + str([dist.mu for dist in self.distributions]) + " and covariances " + str([dist.Sigma for dist in self.distributions]) + "."

class BetaRV:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def pdf(self, x):
        return sp.stats.beta.pdf(x, self.alpha, self.beta)

    def sample(self, N):
        return np.random.beta(self.alpha, self.beta, N)
    
    def __str__(self):
        return "Beta(" + str(self.alpha) + ", " + str(self.beta) + ")"
    
    def __repr__(self):
        return "Beta(" + str(self.alpha) + ", " + str(self.beta) + ")"
    
class LogitNormalRV:

    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma

    def pdf(self, x):
        return 1 / (x * (1 - x)) * sp.stats.norm(self.mu, self.Sigma).pdf(sp.special.logit(x))
    
    def sample(self, N):
        return sp.special.expit(np.random.normal(self.mu, self.Sigma, N))
    
    def __str__(self):
        return "LogitNormal(" + str(self.mu) + ", " + str(self.Sigma) + ")"
    
    def __repr__(self):
        return "LogitNormal(" + str(self.mu) + ", " + str(self.Sigma) + ")"

    def mean(self, Nsamples=10000):
        if self.mu == 0:
            return 0.5
        np.random.seed(0)
        return np.mean(self.sample(Nsamples))
    
    def std(self, Nsamples=10000):
        # estimate second moment using monte carlo
        np.random.seed(0)
        second_moment = np.mean(self.sample(Nsamples)**2)
        return np.sqrt(second_moment - self.mean(Nsamples)**2)
        

def average_normal_dist(dist_arr):
    n = len(dist_arr)
    avg_mu = np.zeros(dist_arr[0].mu.shape)
    avg_Sigma = np.zeros(dist_arr[0].Sigma.shape)
    for dist in dist_arr:
        avg_mu += dist.mu
        avg_Sigma += dist.Sigma
    avg_mu /= n
    avg_Sigma /= n
    return NormalRV(avg_mu, avg_Sigma)
