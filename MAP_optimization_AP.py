import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat

PulseData = np.genfromtxt('Xuedan_PulseDataFriday.csv', delimiter=',')
# print(np.shape(PulseData))

# Load data from .mat with scipy
mat = loadmat('g2_FCS_data2.mat')
auto_corr = mat['auto_corr']
cross_corr = mat['cross_corr']
lagtimes = mat['lagtimes']
xfit_mat = mat['x']
yfit_mat = mat['yfit']

#%% define loss function and relevant parameters

# sparsity constraint parameter
lambda1 = 0.1

Nvars   = 6   # number of fitting parameters

# Lisa / Jian equation
# def func(x, A, B, m, Td, k, Ta):
#     return 1+A*((1-B*x**(m))*1/(1+(x/Td)))-k*np.exp(-x/Ta) # this is the equation I used to get the fits in MATLAB, and almost got there using lmfit but it just kept giving strange results

def lossP(theta,x,y0): # MAP loss function with Poisson noise assumption
    
    w = np.array(theta)
    # f = w[4]**2 + w[0]**2*np.exp(-np.abs(x)/w[1]**2)*( np.exp(-np.abs(x+3*w[2])/w[3]**2) + np.exp(-np.abs(x+2*w[2])/w[3]**2) + np.exp(-np.abs(x+1*w[2])/w[3]**2) + np.exp(-np.abs(x-1*w[2])/w[3]**2) + np.exp(-np.abs(x-2*w[2])/w[3]**2) + np.exp(-np.abs(x-3*w[2])/w[3]**2)) + w[5]**2*np.exp(-np.abs(x)/w[3]**2)
    f = 1 + (w[0]**2) * ((1-(w[1]**2) * x**(2-(w[2]**2))) * 1 / (1+(x/(w[3]**2)))) - w[4]**2 * np.exp(-x/(w[5]**2))

    return np.sum( f -(y0)*np.log( f + 1e-13 ) )  

def lossL(theta,x,y0): # MAP loss function with Gaussian noise (least squares)
    
    w = np.array(theta)
    # f2 = w[4]**2 + w[0]**2*np.exp(-np.abs(x)/w[1]**2)*( np.exp(-np.abs(x+3*w[2])/w[3]**2) + np.exp(-np.abs(x+2*w[2])/w[3]**2) + np.exp(-np.abs(x+1*w[2])/w[3]**2) + np.exp(-np.abs(x-1*w[2])/w[3]**2) + np.exp(-np.abs(x-2*w[2])/w[3]**2) + np.exp(-np.abs(x-3*w[2])/w[3]**2)) + w[5]**2*np.exp(-np.abs(x)/w[3]**2)
    # f2 = 1+w[0]*((1-w[1]*x**(w[2]))*1/(1+(x/w[3])))-w[4]*np.exp(-x/w[5])
    f2 = 1+w[0]**2*((1-w[1]**2*x**(2-(w[2]**2)))*1/(1+(x/w[3]**2)))-w[4]**2*np.exp(-x/w[5]**2)
    return 0.5*np.sum( ( (y0) - f2  )**2 )


def f2(theta,x): # function output (clean)
    
    w = np.array(theta)
    # f2 = w[4]**2 + w[0]**2*np.exp(-np.abs(x)/w[1]**2)*( np.exp(-np.abs(x+3*w[2])/w[3]**2) + np.exp(-np.abs(x+2*w[2])/w[3]**2) + np.exp(-np.abs(x+1*w[2])/w[3]**2) + np.exp(-np.abs(x-1*w[2])/w[3]**2) + np.exp(-np.abs(x-2*w[2])/w[3]**2) + np.exp(-np.abs(x-3*w[2])/w[3]**2)) + w[5]**2*np.exp(-np.abs(x)/w[3]**2)
    # f2 = 1+w[0]*((1-w[1]*x**(w[2]))*1/(1+(x/w[3])))-w[4]*np.exp(-x/w[5])
    # f2 = 1+w[0]**2*((1-w[1]**2*x**(2-(w[2]**2)))*1/(1+(x/w[3]**2)))-w[4]**2*np.exp(-x/w[5]**2)
    f2 = 1 + (w[0]**2) * ((1-(w[1]**2) * x**(2-(w[2]**2))) * 1 / (1+(x/(w[3]**2)))) - w[4]**2 * np.exp(-x/(w[5]**2))
    return f2

def f3Poisson(theta,x,T): # function output (Poisson)

    w = np.array(theta)
    # f3 = w[4]**2 + w[0]**2*np.exp(-np.abs(x)/w[1]**2)*( np.exp(-np.abs(x+3*w[2])/w[3]**2) + np.exp(-np.abs(x+2*w[2])/w[3]**2) + np.exp(-np.abs(x+1*w[2])/w[3]**2) + np.exp(-np.abs(x-1*w[2])/w[3]**2) + np.exp(-np.abs(x-2*w[2])/w[3]**2) + np.exp(-np.abs(x-3*w[2])/w[3]**2)) + w[5]**2*np.exp(-np.abs(x)/w[3]**2)
    # f3 = 1+w[0]*((1-w[1]*x**(w[2]))*1/(1+(x/w[3])))-w[4]*np.exp(-x/w[5])
    f3 = 1+w[0]**2*((1-w[1]**2*x**(2-(w[2]**2)))*1/(1+(x/w[3]**2)))-w[4]**2*np.exp(-x/w[5]**2)
    return np.random.poisson(f3*T  ,size=len(x))

#%% classical optimization parameters

opts = {'maxiter' : 10000,    
#         'maxfun'  : 10000, 
#         'disp' : True,    
#         'full_output': True,  
        'gtol' : 1e-15,
#         'ftol' : 1e-14,
        'eps'  : 1e-15}  # default value.

# minimization
# xTest  = np.array(t50s)
# yTest  = np.array(G2_16min)

x = lagtimes #
y = cross_corr[:,-1] # This is the actual data to be fit
# y = yfit_mat # This is a fit I got and exported from MATLAB to make it easier for python to try and fit

x = x[15:-1]
y = y[15:-1] # Need to cut the really noisy section

xTest = np.array(x)
yTest = np.array(y)

Nruns = 1000

thetaFinalP1 = np.zeros((Nruns,Nvars))
thetaFinalL1 = np.zeros((Nruns,Nvars))
LossFinalP1  = np.zeros(Nruns)
LossFinalL1  = np.zeros(Nruns)

for k in range(Nruns): # perform optimization over multiple random initial conditions

    # guess       = np.random.uniform(0,1,Nvars) # randomize initial guesses
    guess       = np.sqrt([0.31, 6E-01, 4E-08, 3.6E-3, 2.7E-1, 0.1])

    # POISSON REGRESSION
    ResultP1 = minimize(lossP, guess, args=(xTest, yTest), method='Powell', options=opts)
    # LEAST SQUARES REGRESSION
    ResultL1 = minimize(lossL, guess, args=(xTest, yTest), method='Powell', options=opts)

    thetaFinalP1[k,:]   = ResultP1.x
    thetaFinalL1[k,:]   = ResultL1.x
    LossFinalP1[k] = ResultP1.fun
    LossFinalL1[k] = ResultL1.fun

    print(k)

# keep indices of the minimum out of all the trial runs
idxP1 = np.argmin(LossFinalP1)
idxL1 = np.argmin(LossFinalL1)

# FINAL PLOTS

wspace=0.5

# Poisson simulations
# Psim1 = f3Poisson(thetaFinalP1[idxP1,:],xTest,1)

#%% FINAL PLOTS
plt.rc('axes', linewidth=2)
plt.rc('axes',labelsize=20)

fig = plt.figure(dpi=150,figsize=(10,4))
plt.subplots_adjust(wspace=wspace)
ax = plt.subplot(1,2,1)
plt.plot(xTest, yTest, color=(0,0,0.8), linewidth=1.5)
plt.title('$\\bf{(a)}$ Experiment',fontsize=18)
plt.tick_params(labelsize=16)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(xTest,f2(thetaFinalP1[idxP1,:],xTest), color=(0.3,0.3,0.3), linewidth=2.5)
plt.subplots_adjust(wspace=wspace)
plt.tight_layout()
plt.title('$\\bf{(b)}$ Recovery',fontsize=18)
plt.tick_params(labelsize=16)
#
# plt.subplot(3,3,3)
# plt.plot(xTest,f3Poisson(thetaFinalP1[idxP1,:],xTest,1), color=(0.8,0,0), linewidth=1.5)
# plt.title('$\\bf{(c)}$ Poisson Sampling',fontsize=18)
# plt.ylim(0,7)
# plt.tight_layout()
# plt.tick_params(labelsize=16)
# plt.subplots_adjust(wspace=wspace)
#
# plt.subplot(3,3,6)
# plt.plot(xTest,f3Poisson(thetaFinalP1[idxP1,:],xTest,4), color=(0.8,0,0), linewidth=1.5)
# plt.tight_layout()
# plt.ylim(-1,15)
# plt.tick_params(labelsize=16)
# plt.subplots_adjust(wspace=wspace)
#
# plt.subplot(3,3,9)
# plt.plot(xTest,f3Poisson(thetaFinalP1[idxP1,:],xTest,60), color=(0.8,0,0), linewidth=1.5)
# plt.xlabel('$\\tau$ ($\mu$s)')
# plt.ylim(-10,200)
# plt.tight_layout()
# plt.tick_params(labelsize=16)
# plt.subplots_adjust(wspace=wspace)

print('Poisson loss function:',LossFinalP1[idxP1])
