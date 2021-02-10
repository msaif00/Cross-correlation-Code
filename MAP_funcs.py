import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

def find_MAP(x, y, Nruns, Nvars, function, **kwargs):
    opts = {'maxiter': 10000
            #         'maxfun'  : 10000,
            #         'disp' : True,
            #         'full_output': True,
            # 'gtol': 1e-15,
            #         'ftol' : 1e-14,
            # 'eps': 1e-15
                            }  # default value.

    if 'guess' in kwargs:
        input_guess = kwargs['guess']

    if 'randomizer' in kwargs:
        randomizer  = kwargs['randomizer']

    lambda1 = 0.1  # sparsity constraint parameter
    Nruns = Nruns
    Nvars = Nvars  # number of fitting parameters

    thetaFinalP = np.zeros((Nruns, Nvars))
    thetaFinalL = np.zeros((Nruns, Nvars))
    LossFinalP = np.zeros(Nruns)
    LossFinalL = np.zeros(Nruns)

    for k in tqdm(range(Nruns), position=0, leave=True):  # perform optimization over multiple random initial conditions

        if 'guess' in kwargs:
            guess = np.array(input_guess)
        else:
            guess = np.random.uniform(0, 100, Nvars) # randomize initial guesses

        if 'randomizer' in kwargs:
            random_guess = np.random.uniform(0, 1, Nvars)
            new_guess = random_guess * randomizer + guess * (1 - randomizer)
            guess = new_guess

        # print('Initial parameter guess %s:' % guess)
        ResultP1 = minimize(lossP, guess, args=(x, y, function), method='Powell', options=opts)  # POISSON REGRESSION
        ResultL1 = minimize(lossL, guess, args=(x, y, function), method='Powell', options=opts)  # LEAST SQUARES REGRESSION
        thetaFinalP[k, :] = ResultP1.x
        thetaFinalL[k, :] = ResultL1.x
        LossFinalP[k] = ResultP1.fun
        LossFinalL[k] = ResultL1.fun
        # print(k)

    # keep indices of the minimum out of all the trial runs
    idxP = np.argmin(LossFinalP)
    idxL = np.argmin(LossFinalL)

    return thetaFinalP, idxP, thetaFinalL, idxL

def lossP(theta, x, y0, function): # MAP loss function with Poisson noise assumption
    w = np.array(theta)
    f = eval(function)
    return np.sum(f-(y0)*np.log(f+1e-13))

def lossL(theta, x, y0, function): # MAP loss function with Gaussian noise (least squares)
    w = np.array(theta)
    f2 = eval(function)
    return 0.5*np.sum(((y0)-f2)**2)

def f2(theta, x, function): # function output (clean)
    w = np.array(theta)
    f2 = eval(function)
    return f2

def f3Poisson(theta, x, T, function): # function output (Poisson)
    w = np.array(theta)
    f3 = eval(function)
    return  np.random.poisson(f3*T,size=len(x))