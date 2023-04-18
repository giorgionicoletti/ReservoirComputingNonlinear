import numpy as np
from numba import njit, prange

from sklearn import linear_model

from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

import scipy.integrate

import utils

class RateEchoStateNet():
    
    def __init__(self, NE, NI, NInputs, dt, tau_E, tau_I, gamma = 1, max_bias = 1, seed = 42, burnSteps = 10000, input_sparse = 0.8):
        np.random.seed(seed)
        
        self.NE = NE
        self.NI = NI
        self.NInputs = NInputs
        
        self.Win = np.random.randn(self.NInputs, self.NE)
        self.Win[self.Win < input_sparse] = 0
        
        self.dt = dt
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.gamma = gamma
        
        self.plasticity = False
                
        self.WEE = np.random.rand(NE, NE)
        self.WEE[np.diag_indices(NE)] = 0
        
        self.WII = np.random.rand(NI, NI)
        self.WII[np.diag_indices(NI)] = 0

        self.WIE = np.random.rand(NI, NE)
        self.WEI = np.random.rand(NE, NI)

        for W in [self.WIE, self.WII]:
            W /= np.sum(W, axis = 1)[..., None]
            
        for W in [self.WEI, self.WEE]:
            W /= (np.sum(W, axis = 1) + self.Win.sum(axis = 0))[..., None]
            
        self.biases_E = np.random.uniform(0, max_bias, size = self.NE)
        self.biases_I = np.random.uniform(0, max_bias, size = self.NI)
        
        if burnSteps is not None:
            E0 = np.zeros(NE)
            I0 = np.zeros(NI)

            EBurn, IBurn = self.run_recurrent(np.zeros(burnSteps), E0, I0)
            self.E0 = EBurn[-1]
            self.I0 = IBurn[-1]

            EBurn = None
            IBurn = None
        else:
            self.E0 = np.random.uniform(0, 1, size = NE)
            self.I0 = np.random.uniform(0, 1, size = NI)
            
        self._set_dynamics()
        
        self.generate_recurrent = True
            
    def _set_dynamics(self):
        self.params_dyn = (self.dt, self.Win,
                           self.WEE, self.WEI, self.WIE, self.WII, 
                           self.tau_E, self.tau_I, self.biases_E, self.biases_I, self.gamma)
            
    def _set_plasticity(self, eta_EE, eta_EI, eta_IE, eta_II, rho_E, rho_I):
        self.eta_EE = eta_EE
        self.eta_II = eta_II
        self.eta_EI = eta_EI
        self.eta_IE = eta_IE
        
        self.rho_E = rho_E
        self.rho_I = rho_I
        
        self.plasticity = True
        
        self.params_plast = (self.eta_EE, self.eta_EI, self.eta_IE, self.eta_II, self.rho_E, self.rho_I)
        
    def run_plasticity(self, u, log = True):
        if not self.plasticity:
            raise utils.NotInitialized("Plasticity parameters have not been initialized yet")
            
        args = (u, self.E0, self.I0,
                *self.params_plast, *self.params_dyn)
                        
        results = utils.plasticity_numba(*args)

        self.E_plasticity, self.I_plasticity = results
        
    def run_recurrent(self, u, E0 = None, I0 = None):
        assert u.shape[1] == self.NInputs
        
        if E0 is None:
            E0 = self.E0
        if I0 is None:
            I0 = self.I0
        
        results = utils.recurrent_numba(u, E0, I0, *self.params_dyn)

        return results
    
    def train_model(self, u, initTraining, max_iter = 1000, alpha = 1e-4):
        regression = linear_model.Ridge(alpha = alpha, max_iter = max_iter, fit_intercept = False)
        
        if self.generate_recurrent == True:
            self.E_rec, self.I_rec = self.run_recurrent(u)
            self.generate_recurrent = False
        
        self.target = u[initTraining + 1:].T
        self.inputs = u[initTraining:-1]
        X = self.E_rec[initTraining + 1:, :]

        self.regression = regression.fit(X, self.target.T)
        self.Wout = self.regression.coef_
        
        trained_output = np.dot(self.Wout, X.T)
        
        fix, axs = plt.subplots(figsize = (20,10), nrows = 3)
        for i in range(3):
            axs[i].plot(trained_output[i], c = 'orangered', lw = 2)
            axs[i].plot(self.target[i], ls = '--', c = 'navy', lw = 1)
            
        MSE = np.sqrt(np.mean((trained_output - self.target)**2, axis = 1)).mean()
        
        return MSE, trained_output
    
    def echo_state(self, u, nLoops, idx_start, idx_echo):
        output_sequence = np.zeros((nLoops + 1 + idx_echo, self.NInputs))
        
        E = np.zeros((nLoops + 1 + idx_echo, self.NE), dtype = np.float64)
        I = np.zeros((nLoops + 1 + idx_echo, self.NI), dtype = np.float64)
        
        E[0] = self.E_rec[idx_start]
        I[0] = self.I_rec[idx_start]
        output_sequence[0] = self.predict(E[0])
        
        for i in tqdm(range(idx_echo)):
            E[i+1], I[i+1] = self.dynamical_step(E[i], I[i], u[idx_start + i])
            output_sequence[i+1] = self.predict(E[i+1])
        
        for i in tqdm(range(idx_echo, nLoops + idx_echo)):
            E[i+1], I[i+1] = self.dynamical_step(E[i], I[i], output_sequence[i])
            output_sequence[i+1] = self.predict(E[i+1])
        
        return E, I, output_sequence
            
    def predict(self, E):
        return np.dot(self.Wout, E)
    
    def return_input(self, ut):        
        return np.dot(ut, self.Win)
    
    def dynamical_step(self, E, I, ut):
        new_E = E + self.dt/self.tau_E*(-E + self.gamma*utils.ReLU(np.dot(self.WEE, E) - np.dot(self.WEI, I) + self.return_input(ut) - self.biases_E))
        new_I = I + self.dt/self.tau_I*(-I + utils.ReLU(np.dot(self.WIE, E) - np.dot(self.WII, I) - self.biases_I))
        
        return new_E, new_I