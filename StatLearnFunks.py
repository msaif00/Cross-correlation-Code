import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from scipy.optimize import minimize, optimize

class MaximumAPosteriori(object):
    """
    Object that is filled with parameter arrays, indices for best optimal parameters, and fits with optimal parameters
    from MAP estimation after running find_MAP
    """
    def __init__(self):
        self.idxP = []
        self.idxL = []
        self.idxPL = []

        self.thetaFinalP = []
        self.thetaFinalL = []
        self.thetaFinalPL = []

        self.theta_P = []
        self.theta_L = []
        self.theta_PL = []

        self.fit_P = []
        self.fit_L = []
        self.fit_PL = []

        self.LossFinalP = []
        self.LossFinalL = []
        self.LossFinalPL = []

    def find_MAP(self, x, y, Nruns, Nvars, function, **kwargs):
        print('testing')
        """
        Written by Cristian L. Cortes, Argonne National Lab, 2020
        Modified by Andrew H. Proppe, Massachusetts Institute of Technology, 2021

        Inputs:
        x = x data points, numpy array
        y = y data points, numpy array
        Nruns = number of optimizations with (optionally) different initial guesses
        Nvars = number of parameters used in the fit equation
        guess = optional input array of initial guess for optimization
        randomizer = value between 0 and 1 that blends input guess with random guess (0 for 100% input guess, 1 for 100% random guess)
                     or if bounds are given, use 'bounded' to generate random guesses between lower and upper bounds
        bounds = tuple of lower and upper bounds (e.g. ((0,1),(10,100))) that work with certain scipy minimization methods
        random_range = decade of range that is used for random guesses (e.g. random guess between 0 and 10^(random_range))
        """
        # print('test')
        np.random.seed(12345)

        opts = {'maxiter': 10000,
                #         'maxfun'  : 10000,
                #         'disp' : True,
                # 'full_output': True,
                # 'ftol': 1e-15,
                #         'ftol' : 1e-14,
                # 'eps': 1e-15
                                }  # default value.

        if 'guess' in kwargs:
            input_guess = kwargs['guess']

        if 'randomizer' in kwargs:
            randomizer  = kwargs['randomizer']

        if 'random_range' in kwargs:
            random_range  = kwargs['random_range']

        if 'bounds' in kwargs:
            bounds = kwargs['bounds']
            lb = np.array(bounds[0])
            ub = np.array(bounds[1])
            bds = np.array((lb, ub)).T
            # bds = 'hello'
            bds = tuple(map(tuple, bds))

        lambda1 = 0.1  # sparsity constraint parameter
        Nruns = Nruns
        Nvars = Nvars  # number of fitting parameters

        thetaFinalP = np.zeros((Nruns, Nvars))
        thetaFinalL = np.zeros((Nruns, Nvars))
        thetaFinalPL = np.zeros((Nruns, Nvars))

        LossFinalP = np.zeros(Nruns)
        LossFinalL = np.zeros(Nruns)
        LossFinalPL = np.zeros(Nruns)

        for k in tqdm(range(Nruns), position=0, leave=True):  # perform optimization over multiple random initial conditions

            if 'guess' in kwargs:
                guess = np.array(input_guess)
            else:
                if 'random_range' in kwargs:
                    guess = np.random.uniform(0, 10**random_range, Nvars)  # randomize initial guesses
                else:
                    guess = np.random.uniform(0, 100, Nvars) # randomize initial guesses

            if 'randomizer' in kwargs and 'guess' in kwargs: # Use randomizer on input guess
                if randomizer == 'bounded' and 'bounds' in kwargs:
                    guess = np.random.uniform(0.01, 0.99, Nvars) * (ub - lb) + lb
                else:
                    if 'random_range' in kwargs:
                        random_guess = np.random.uniform(0, 10**random_range, Nvars) # Specify decade for range of random array
                    else:
                        random_guess = np.random.uniform(0, 1, Nvars)
                    new_guess = random_guess * randomizer + guess * (1 - randomizer)
                    guess = new_guess

            if 'bounds' in kwargs:
                ResultP1 = minimize(lossP, guess, args=(x, y, function), method='Powell', options=opts, bounds=bds)  # POISSON REGRESSION
                ResultL1 = minimize(lossL, guess, args=(x, y, function), method='Powell', options=opts, bounds=bds)  # LEAST SQUARES REGRESSION
                ResultPL1 = minimize(lossPL, guess, args=(x, y, function), method='Powell', options=opts, bounds=bds) # HYBRID REGRESSION

            else:
                ResultP1 = minimize(lossP, guess, args=(x, y, function), method='Powell', options=opts)  # POISSON REGRESSION
                ResultL1 = minimize(lossL, guess, args=(x, y, function), method='Powell', options=opts)  # LEAST SQUARES REGRESSION
                ResultPL1 = minimize(lossPL, guess, args=(x, y, function), method='Powell', options=opts) # HYBRID REGRESSION

            thetaFinalP[k, :] = ResultP1.x
            thetaFinalL[k, :] = ResultL1.x
            thetaFinalPL[k, :] = ResultPL1.x

            LossFinalP[k] = ResultP1.fun
            LossFinalL[k] = ResultL1.fun
            LossFinalPL[k] = ResultPL1.fun

        thetaFinalP = thetaFinalP
        thetaFinalL = thetaFinalL
        thetaFinalPL = thetaFinalPL

        # keep indices of the minimum out of all the trial runs
        idxP = np.argmin(LossFinalP)
        idxL = np.argmin(LossFinalL)
        idxPL = np.argmin(LossFinalPL)

        # fits from best MAP parameters
        map_fit_P = f2(thetaFinalP[idxP, :], x, function)  # Poisson
        map_fit_L = f2(thetaFinalL[idxL, :], x, function)  # Gaussian
        map_fit_PL = f2(thetaFinalPL[idxPL, :], x, function)  # Hybrid
        theta_mapP = thetaFinalP[idxP, :]  # best parameters from MAP (Poisson)
        theta_mapL = thetaFinalL[idxL, :]  # best parameters from MAP (Gaussian)
        theta_mapPL = thetaFinalL[idxPL, :]  # best parameters from MAP (Hybrid)

        self.idxP = idxP
        self.idxL = idxL
        self.idxPL = idxPL

        self.thetaFinalP = thetaFinalP
        self.thetaFinalL = thetaFinalL
        self.thetaFinalPL = thetaFinalPL

        self.theta_P = theta_mapP
        self.theta_L = theta_mapL
        self.theta_PL = theta_mapPL

        self.fit_P = map_fit_P
        self.fit_L = map_fit_L
        self.fit_PL = map_fit_PL

        self.LossFinalP = LossFinalP
        self.LossFinalL = LossFinalL
        self.LossFinalPL = LossFinalPL

        return self

def lossP(theta, x, y0, function): # MAP loss function with Poisson noise assumption
    w = np.array(theta)
    f = eval(function)
    return np.sum(f-(y0)*np.log(f+1e-13))

def lossL(theta, x, y0, function): # MAP loss function with Gaussian noise (least squares)
    w = np.array(theta)
    f2 = eval(function)
    return 0.5*np.sum(((y0)-f2)**2)

def lossPL(theta, x, y0, function): # MAP loss function with Gaussian + Poisson noise
    w = np.array(theta)
    f = eval(function)
    return np.sum(f-(y0)*np.log(f+1e-13)) + 0.5*np.sum(((y0)-f)**2)

def f2(theta, x, function): # function output (clean)
    w = np.array(theta)
    f2 = eval(function)
    return f2

def f3Poisson(theta, x, T, function): # function output (Poisson)
    w = np.array(theta)
    f3 = eval(function)
    return np.random.poisson(f3*T, size=len(x))

def Poisson_noise_g2(theta, x, T, function): # function output with Poisson noise in log spaced bins
    w = np.array(theta)
    f3 = eval(function)
    return np.random.poisson(f3*x*T, size=len(x))/(x*T)

    # def pairplot_divergence(trace, vars, ax=None, divergence=True, color="C3", divergence_color="C2"):
    #     var1 = trace.get_values(varname=vars[0], combine=True)
    #     var2 = trace.get_values(varname=vars[1], combine=True)
    #     if not ax:
    #         _, ax = plt.subplots(1, 1, figsize=(10, 5))
    #     ax.plot(var1, var2, "o", color=color, alpha=0.5)
    #     if divergence:
    #         divergent = trace["diverging"]
    #         ax.plot(var1[divergent], var2[divergent], "o", color=divergence_color)
    #     ax.set_xlabel(vars[0])
    #     ax.set_ylabel(vars[1])
    #     ax.set_title("scatter plot between %s and %s" % (vars[0], vars[1]))
    #     return ax

def pairplot_divergence(trace, vars, ax=None, divergence=True, color="C3", divergence_color="C2"):
    """
    Adapted from PyMC3 example for pairplots between only two vars (https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html)
    :param trace: input trace from pm.sample
    :param vars:
    :param ax:
    :param divergence: plot divergent sampled points
    :return:
    """
    if not ax:
        _, ax = plt.subplots(len(vars), len(vars), figsize=(12, 8))

    for a in range(len(vars)):
        for b in range(len(vars)):
            var1 = trace.get_values(varname=vars[a], combine=True)
            var2 = trace.get_values(varname=vars[b], combine=True)
            ax[a,b].plot(var1, var2, "o", color=color, alpha=0.5)
            if divergence:
                divergent = trace["diverging"]
                ax[a,b].plot(var1[divergent], var2[divergent], "o", color=divergence_color)
            ax[a,b].set_xlabel(vars[a])
            ax[a,b].set_ylabel(vars[b])
            # ax[a,b].set_title("scatter plot between %s and %s" % (vars[a], vars[b]))

    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    return ax

def report_trace(trace, varname, log_on_off):
    """
    # A small wrapper function for displaying the MCMC sampler diagnostics; adapted from PyMC3 example (https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html)
    :param trace: input trace from pm.sample
    :param varname: name of variable (string)
    :param log_on_off: plot log of variable's value (true / false)
    :return:
    """

    # plot the estimate for the mean of log(τ) cumulating mean
    if log_on_off:
        var = np.log(trace[varname])
    else:
        var = (trace[varname])

    mvar = [np.mean(var[:i]) for i in np.arange(1, len(var))]
    plt.figure(figsize=(10, 4))
    plt.plot(mvar, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("MCMC mean of %s" % varname)
    plt.title("MCMC estimation of %s " % varname)
    plt.show()

    # plot the trace of log(tau)
    # pm.traceplot({varname: trace.get_values(varname=varname, combine=False)})

    print("Starting value: %s" % var[0].round(2))

    # display the total number and percentage of divergent
    divergent = trace["diverging"]
    print("Number of Divergent %d" % divergent.nonzero()[0].size)
    divperc = divergent.nonzero()[0].size / len(trace) * 100
    print("Percentage of Divergent %.1f" % divperc)


# divergent_point = defaultdict(list)
# chain_warn = trace.report._chain_warnings
# for i in range(len(chain_warn)):
#     for warning_ in chain_warn[i]:
#         if warning_.step is not None and warning_.extra is not None:
#             for RV in model.free_RVs:
#                 para_name = RV.name
#                 divergent_point[para_name].append(warning_.extra[para_name])
#
# for RV in model.free_RVs:
#     para_name = RV.name
#     divergent_point[para_name] = np.asarray(divergent_point[para_name])
# ii = 5
#
#
# tau_log_d = divergent_point["tau_log__"]
# theta0_d = divergent_point["theta"][:, ii]
# Ndiv_recorded = len(tau_log_d)
#
# _, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
#
# pairplot_divergence(trace, ax=ax[0], color="C7", divergence_color="C2")
#
# plt.title("scatter plot between log(tau) and theta[0]")
#
# pairplot_divergence(trace, ax=ax[1], color="C7", divergence_color="C2")
#
# theta_trace = trace["theta"]
# theta0 = theta_trace[:, 0]
#
# ax[1].plot(
#     [theta0[divergent == 1][:Ndiv_recorded], theta0_d],
#     [logtau[divergent == 1][:Ndiv_recorded], tau_log_d],
#     "k-",
#     alpha=0.5,
# )
#
# ax[1].scatter(
#     theta0_d, tau_log_d, color="C3", label="Location of Energy error (start location of leapfrog)"
# )
#
# plt.title("scatter plot between log(tau) and theta[0]")
# plt.legend();