import matplotlib.pyplot as plt
import numpy as np
from nsimpkg.random_variables import average_normal_dist, BetaRV
from nsimpkg.mcsim import rho

def plotter(distributions, pi, title, mix=False, alpha=0.2, red_lines=True):

    Niter = len(distributions[0])
    distributions_t = np.array(distributions).T
    average_distributions = [average_normal_dist(d) for d in distributions_t]
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    axs[0].plot([d.mu[0] for d in average_distributions], color="blue", label="Average over runs", linewidth=2)
    axs[1].plot([d.mu[1] for d in average_distributions], color="blue", label="Average over runs", linewidth=2)
    for i, distribution_list in enumerate(distributions):
        if i == 0:
            axs[0].plot([d.mu[0] for d in distribution_list], color="black", alpha=alpha, label="Single run", linewidth=1)
            axs[1].plot([d.mu[1] for d in distribution_list], color="black", alpha=alpha, label="Single run", linewidth=1)
        else:
            mu_1 = [d.mu[0] for d in distribution_list]
            mu_2 = [d.mu[1] for d in distribution_list]
            axs[0].plot(mu_1, color="black", alpha=alpha)
            axs[1].plot(mu_2, color="black", alpha=alpha)
    if mix:
        if red_lines:
            axs[0].hlines(pi.avg_dist.mu[0], 0, Niter, label="Average value", color="red", linestyle="--", linewidth=2)
            axs[1].hlines(pi.avg_dist.mu[1], 0, Niter, label="Average value", color="red", linestyle="--", linewidth=2)
        true = pi.avg_dist.Sigma
    else:
        axs[0].hlines(pi.mu[0], 0, Niter, label="True value", color="red", linestyle="--")
        axs[1].hlines(pi.mu[1], 0, Niter, label="True value", color="red", linestyle="--")
        true = pi.Sigma
    axs[0].set_xlabel("Iteration number (log scale)")
    axs[0].set_ylabel("Numerical value")
    axs[0].set_title("$(\mu_k)_1$", fontsize=16)
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[1].set_title("$(\mu_k)_2$", fontsize=16)

    true00 = true[0][0]
    true11 = true[1][1]
    true01 = true[0][1]
    true10 = true[1][0]
    sigma00 = [d.Sigma[0][0] for d in average_distributions]
    sigma10 = [d.Sigma[1][0] for d in average_distributions]
    sigma01 = [d.Sigma[0][1] for d in average_distributions]
    sigma11 = [d.Sigma[1][1] for d in average_distributions]
    axs[2].plot(sigma00, color="blue", label="Average over runs", linewidth=2)
    axs[3].plot(sigma10, color="blue", label="Average over runs", linewidth=2)
    axs[4].plot(sigma01, color="blue", label="Average over runs", linewidth=2)
    axs[5].plot(sigma11, color="blue", label="Average over runs", linewidth=2)
    for i, distribution_list in enumerate(distributions):
        if i == 0:
            sigma00 = [d.Sigma[0][0] for d in distribution_list]
            sigma10 = [d.Sigma[1][0] for d in distribution_list]
            sigma01 = [d.Sigma[0][1] for d in distribution_list]
            sigma11 = [d.Sigma[1][1] for d in distribution_list]
            axs[2].plot(sigma00, color="black", alpha=alpha, label="Single run", linewidth=1)
            axs[3].plot(sigma10, color="black", alpha=alpha, label="Single run", linewidth=1)
            axs[4].plot(sigma01, color="black", alpha=alpha, label="Single run", linewidth=1)
            axs[5].plot(sigma11, color="black", alpha=alpha, label="Single run", linewidth=1)
        sigma00 = [d.Sigma[0][0] for d in distribution_list]
        sigma10 = [d.Sigma[1][0] for d in distribution_list]
        sigma01 = [d.Sigma[0][1] for d in distribution_list]
        sigma11 = [d.Sigma[1][1] for d in distribution_list]
        axs[2].plot(sigma00, color="black", alpha=alpha, linewidth=1)
        axs[3].plot(sigma10, color="black", alpha=alpha, linewidth=1)
        axs[4].plot(sigma01, color="black", alpha=alpha, linewidth=1)
        axs[5].plot(sigma11, color="black", alpha=alpha, linewidth=1)
    
    if red_lines:
        axs[2].hlines(y=true00, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value" if not mix else "Average value")
        axs[3].hlines(y=true01, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value" if not mix else "Average value")
        axs[4].hlines(y=true10, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value" if not mix else "Average value")
        axs[5].hlines(y=true11, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value" if not mix else "Average value")
    axs[2].set_title('$(\Sigma_k)_{1,1}$', fontsize=16)
    axs[3].set_title('$(\Sigma_k)_{1,2}$', fontsize=16)
    axs[4].set_title('$(\Sigma_k)_{2,1}$', fontsize=16)
    axs[5].set_title('$(\Sigma_k)_{2,2}$', fontsize=16)
    for ax in axs.flat:
        ax.set_xscale('log')
        # turn grid on
    axs[0].set_xlabel("Iteration number (log scale)", fontsize=14)
    axs[0].set_ylabel("Numerical value", fontsize=14)
    axs[0].legend(fontsize=11.5)
    # make common legend for all subplots
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()

def plotter_means(distributions, pi, title, mix=False, alpha=0.2):

    Niter = len(distributions[0])
    distributions_t = np.array(distributions).T
    average_distributions = [average_normal_dist(d) for d in distributions_t]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot([i for i in range(1, Niter+1)], [d.mu[0] for d in average_distributions], color="black", label="Average over runs", linewidth=4)
    axs[1].plot([i for i in range(1, Niter+1)], [d.mu[1] for d in average_distributions], color="black", label="Average over runs", linewidth=4)
    for i, distribution_list in enumerate(distributions):
        if i == 0:
            axs[0].plot([i for i in range(1, Niter+1)] ,[d.mu[0] for d in distribution_list], color="black", alpha=alpha, label="Single run", linewidth=2)
            axs[1].plot([i for i in range(1, Niter+1)] ,[d.mu[1] for d in distribution_list], color="black", alpha=alpha, label="Single run", linewidth=2)
        else:
            mu_1 = [d.mu[0] for d in distribution_list]
            mu_2 = [d.mu[1] for d in distribution_list]
            axs[0].plot([i for i in range(1, Niter+1)] ,mu_1, color="black", alpha=alpha)
            axs[1].plot([i for i in range(1, Niter+1)] ,mu_2, color="black", alpha=alpha)
    if mix:
        axs[0].hlines(pi.avg_dist.mu[0], 0, Niter, label="Average value", color="blue", linestyle="--", linewidth=4)
        axs[1].hlines(pi.avg_dist.mu[1], 0, Niter, label="Average value", color="blue", linestyle="--", linewidth=4)
        true = pi.avg_dist.Sigma
    else:
        axs[0].hlines(pi.mu[0], 0, Niter+10000, label="True value", color="blue", linestyle="--", linewidth=3)
        axs[1].hlines(pi.mu[1], 0, Niter+10000, label="True value", color="blue", linestyle="--", linewidth=3)
        true = pi.Sigma
    axs[0].set_xlabel("Iteration number (log scale)")
    axs[0].set_ylabel("Numerical value")
    axs[0].set_title("$(\mu_k)_1$", fontsize=18)
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[1].set_title("$(\mu_k)_2$", fontsize=18)
    for ax in axs.flat:
        ax.set_xscale('log')
        # turn grid on
    axs[0].set_xlabel("Iteration number (log scale)", fontsize=16)
    axs[0].set_ylabel("Numerical value", fontsize=16)
    axs[0].legend(fontsize=14)
    # increase tick size
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    # set xlim
    axs[0].set_xlim(1, Niter+10000)
    axs[1].set_xlim(1, Niter+10000)
    # make common legend for all subplots
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()

def plot_rho(distributions, pi, title, Nsamples=1000, average=False, ylog=False, alpha=0.2):
    if average:
        distributions_t = np.array(distributions).T
        average_distributions = [average_normal_dist(d) for d in distributions_t]
        rhos = [rho(pi, d, Nsamples=Nsamples) for d in average_distributions]
        plt.plot(rhos, color="black")
        plt.xlabel("Iteration number (log scale)")
        if ylog:
            plt.yscale("log")
            plt.ylabel("Log numerical value")
        else:
            plt.ylabel("Numerical value")
        plt.title(title)
    else:
        for experiment in distributions:
            rhos = [rho(pi, d, Nsamples=Nsamples) for d in experiment]
            plt.plot(rhos, color="black", alpha=alpha)
        plt.xlabel("Iteration number (log scale)")
        if ylog:
            plt.yscale("log")
            plt.ylabel("Log numerical value")
        else:
            plt.ylabel("Numerical value")
        plt.title(title)

def fill_z(p, npoints=100, xlim=[-20, 20], ylim=[-20, 20]):
    xmin, xmax = xlim
    ymin, ymax = ylim
    X = np.linspace(xmin, xmax, npoints)
    Y = np.linspace(ymin, ymax, npoints)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((npoints, npoints))
    for i in range(npoints):
        for j in range(npoints):
            Z[i,j] = p.pdf(np.array([X[i,j], Y[i,j]]))
    return X, Y, Z

def plot_contours(distributions, pi, title, mix=False):
    Niter = len(distributions[0])
    distributions_t = np.array(distributions).T
    average_distributions = [average_normal_dist(d) for d in distributions_t]

    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    #axs[0,0].scatter(out_sq[0,:], out_sq[1,:], s=0.1, c='black', label="Samples from the target distribution", zorder=-5)
    #axs[0,0].scatter(in_sq[0,:], in_sq[1,:], s=0.1, c='red', label="Samples in the unit square", zorder=-5)
    X, Y, Z_pi = fill_z(pi)
    axs[-1].contourf(X, Y, Z_pi, levels=10, cmap="Greys", zorder=-10)
    axs[-1].fill_between([-1, 1], [-1, -1], [1, 1], color='red', zorder=5, alpha=0.7, label="Area $D$")
    if not mix:
        pi.construct_ellipse(1, axs[-1], "blue", label="68% confidence region")
        pi.construct_ellipse(2, axs[-1], "green", label="95% confidence region")
        pi.construct_ellipse(3, axs[-1], "red", label="99.7% confidence region")
    #axs[-1].legend(fontsize=6)
    axs[-1].set_xlim(-20, 20)
    axs[-1].set_ylim(-20, 20)
    axs[-1].set_title("Target distribution $\pi$", fontsize=14)

    # get log interval
    dist_to_get = [int(np.exp(i)) for i in np.linspace(0, np.log(Niter), 5)]
    dist_to_get[0] = 0
    dist_to_get[-1] = -1

    remaining_axs = axs[:-1]

    for i, ax in zip(dist_to_get, remaining_axs):
        #ax.scatter(out_sq[0,:], out_sq[1,:], s=0.1, c='black', label="Samples from the target distribution", zorder=-5)
        ax.fill_between([-1, 1], [-1, -1], [1, 1], color='red', zorder=5, alpha=0.7, label="Area $D$")
        X, Y, Z_d = fill_z(average_distributions[i], xlim=[-22, 22], ylim=[-22, 22])
        ax.contourf(X, Y, Z_d, levels=10, cmap="Greys", zorder=-10)

        average_distributions[i].construct_ellipse(3, ax, "red", label="99.7% confidence region")
        average_distributions[i].construct_ellipse(2, ax, "green" , label="95% confidence region")
        average_distributions[i].construct_ellipse(1, ax, "blue", label="68% confidence region")
        ax.set_ylim(-22, 22)
        ax.set_xlim(-22, 22)
        #ax.legend(fontsize=6)
        # disable x and y ticks
        # ax.set_xticks([])
        ax.set_yticks([-20, -10, 0, 10, 20])
        j = i if i != -1 else Niter-1
        ax_title = "Proposal at iteration {}".format(j)
        ax.set_title(ax_title, fontsize=14)
    axs[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()

# define function to plot parameters over time
def plot_params_beta(experiment_distributions, pi, title, Nsamples = 10000, alpha=0.2, xlog=False):
    Niter = len(experiment_distributions[0])
    alphas =  np.array([np.array([dist.alpha for dist in experiment]) for experiment in experiment_distributions])
    betas =  np.array([np.array([dist.beta for dist in experiment]) for experiment in experiment_distributions])
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.2))
    pi_mean = pi.mean(Nsamples) if pi.mu != 0 else 0.5 
    pi_std = pi.std(Nsamples)
    all_means = []
    all_stds = []
    for i, alpha_beta in enumerate(zip(alphas, betas)):
        alpha_experiment, beta_experiment = alpha_beta
        axs[0].plot(alpha_experiment, color="black", alpha=alpha, linewidth=1)
        axs[1].plot(beta_experiment, color="black", alpha=alpha, linewidth=1)
        means = alpha_experiment / (alpha_experiment + beta_experiment)
        stds = np.sqrt(alpha_experiment * beta_experiment / ((alpha_experiment + beta_experiment)**2 * (alpha_experiment + beta_experiment + 1)))
        all_means.append(means)
        all_stds.append(stds)
        if i == 0:
            axs[2].plot(means, color="black", alpha=alpha, label="Single run", linewidth=1)
            axs[3].plot(stds, color="black", alpha=alpha, label="Single run", linewidth=1)
        axs[2].plot(means, color="black", alpha=alpha)
        axs[3].plot(stds, color="black", alpha=alpha)
    average_alphas = np.mean(alphas, axis=0)
    average_betas = np.mean(betas, axis=0)
    average_means = np.mean(all_means, axis=0)
    average_stds = np.mean(all_stds, axis=0)
    axs[0].plot(average_alphas, color="blue", linewidth=2, alpha=0.4)
    axs[1].plot(average_betas, color="blue", linewidth=2, alpha=0.4)
    axs[0].set_title("$\\alpha$", fontsize=11.2)
    axs[1].set_title("$\\beta$", fontsize=11.2)
    if xlog:
        for i, ax in enumerate(axs):
            ax.set_xscale("log")
    axs[0].set_xlabel("Iteration number")
    axs[0].set_ylabel("Value")
    axs[2].plot(average_means, color="blue", linewidth=2, label="Average over all runs", alpha=1)
    axs[3].plot(average_stds, color="blue", linewidth=2, label="Average over all runs", alpha=0.4)
    axs[2].hlines(pi_mean, 0, Niter, color="red", linestyle="--", label="True value", linewidth=0.7)
    axs[3].hlines(pi_std, 0, Niter, color="red", linestyle="--", label="True value", linewidth=1)
    axs[2].set_title("Mean", fontsize=11.2)
    axs[3].set_title("Standard deviation", fontsize=11.2)
    # move legend patches to first axis
    handles, labels = axs[2].get_legend_handles_labels()
    axs[0].legend(handles, labels)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

def plot_iters_beta(experiment_distributions, pi, title):
    Niter = len(experiment_distributions[0])
    grid = np.linspace(0.001, 0.999, Niter)
    alphas =  np.array([np.array([dist.alpha for dist in experiment]) for experiment in experiment_distributions])
    betas =  np.array([np.array([dist.beta for dist in experiment]) for experiment in experiment_distributions])
    average_alphas = np.mean(alphas, axis=0)
    average_betas = np.mean(betas, axis=0)
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    proposal_to_get = np.logspace(0, np.log10(Niter), 6, dtype=int)
    proposal_to_get[proposal_to_get==1] = 0
    for i, proposal in enumerate(proposal_to_get):
        ax = axs[i]
        alpha_plot = average_alphas[proposal]
        beta_plot = average_betas[proposal]
        proposal_plot = BetaRV(alpha_plot, beta_plot)
        proposal_plot_pdf = proposal_plot.pdf(grid)
        ax.plot(grid, proposal_plot_pdf, color="red", label="Proposal distribution")
        ax.plot(grid, pi.pdf(grid), color="black", label="Target distribution")
        ax.set_title("Iteration {}".format(proposal), fontsize=14)
        ax.scatter([0.25, 0.75], [0,0], marker="|", color="blue", zorder=10000, linewidth=2)
        ax.plot([0.25, 0.75], [0,0], color="blue", zorder=1000, linewidth=2, label="Interval $D$")
        ax.set_ylim(0, 1.7)
        ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
        if i== 0:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$p(x)$")
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()

def plot_mse(results_list, GT, title, xlog = False, ylog=False):
    Niter = len(results_list[0])
    results = np.array(results_list)
    mses = np.mean((results-GT)**2, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot([i for i in range(1, Niter+1)], mses, color="black", linewidth=1)
    plt.title(title, fontsize=20)
    if xlog:
        plt.xscale("log")
        plt.xlabel("Iteration number (log scale)", fontsize=16)
    else:
        plt.xlabel("Iteration number", fontsize=16)
    if ylog:
        plt.yscale("log")
        plt.ylabel("MSE (log scale)", fontsize=16)
    else:
        plt.ylabel("MSE", fontsize=16)
    # increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()