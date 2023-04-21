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
    """
    Class for a rate-based Echo State Network, which uses a Wilson-Cowan-like
    model for the dynamics of the neurons.

    The network is initialized with the given parameters, and then it can be
    trained with the given input and output data, or it can be used to generate
    a recurrent sequence in the echo state.
    """
    def __init__(self, NE, NI, NInputs, dt, tau_E, tau_I,
                 gamma = 1, max_bias = 1, seed = 42, burnSteps = 10000, input_sparse = 0.8,
                 method = 'rk4', nonlinearity = utils.ReLU, args_nonlin = ()):
        
        """
        Initializes the network with the given parameters.

        Parameters
        ----------
        NE : int
            Number of excitatory neurons.
        NI : int
            Number of inhibitory neurons.
        NInputs : int
            Dimension of the input space.
        dt : float
            Time step.
        tau_E : float
            Time constant of the excitatory neurons.
        tau_I : float
            Time constant of the inhibitory neurons.
        gamma : float, optional
            Gain factor of the excitatory population. The default is 1.
            Not really needed.
        max_bias : float, optional
            Maximum value of the bias, which are randomly generated between 0 and max_bias.
            The default is 1. If None, no bias is added.
        seed : int, optional
            Seed for the random number generator. The default is 42.
        burnSteps : int, optional
            Number of steps to initialize the network to forget the initial conditions.
            The default is 10000.
        input_sparse : float, optional
            Sparsity of the input weights. The default is 0.8.
        method : str, optional
            Integration method. The default is 'rk4', but 'euler' is also available.
        nonlinearity : function, optional
            Nonlinearity to be used. The default is ReLU. The function must be compiled with numba.
        args_nonlin : tuple, optional
            Arguments of the nonlinearity. The default is ().
        """

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

        self.method = method

        if method == 'rk4':
            self.dynamical_step = utils.rk4_step
        elif method == 'euler':
            self.dynamical_step = utils.euler_step
        else:
            raise utils.NotImplemented("Method must be either 'rk4' or 'euler', no other integrations steps are implemented yet.")

        self.dynamical_step.recompile()

        self.nonlin = nonlinearity
        self.args_nonlin = args_nonlin
        
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
        
        if max_bias is not None:
            self.biases_E = np.random.uniform(0, max_bias, size = self.NE)
            self.biases_I = np.random.uniform(0, max_bias, size = self.NI)
        else:
            self.biases_E = np.zeros(self.NE)
            self.biases_I = np.zeros(self.NI)
        
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
            
    def _set_dynamics(self, dynamical_step = None, nonlinearity = None, args_nonlin = None):
        """
        Set the dynamics of the network in a tuple of parameters.

        Parameters
        ----------
        dynamical_step : function, optional
            Function to be used to integrate the dynamics.
            The default is None, in which case the default function is used.
        nonlinearity : function, optional
            Nonlinearity to be used. The default is None, in which case the default nonlinearity is used.
        args_nonlin : tuple, optional
            Arguments of the nonlinearity. The default is None, in which case the default arguments are used.
        """
        if dynamical_step is not None:
            self.dynamical_step = dynamical_step
        if nonlinearity is not None:
            self.nonlin = nonlinearity
        if args_nonlin is not None:
            self.args_nonlin = args_nonlin

        self.params_dyn = (self.dt, self.tau_E, self.tau_I,
                           self.Win, self.WEE, self.WEI, self.WIE, self.WII, 
                           self.biases_E, self.biases_I, self.gamma,
                           self.nonlin, self.args_nonlin, self.dynamical_step)
            
    def _set_plasticity(self, eta_EE, eta_EI, eta_IE, eta_II, rho_E, rho_I):
        """
        Set the parameters of the plasticity in a tuple of parameters.

        Parameters
        ----------
        eta_EE : float
            Learning rate for excitatory-excitatory connections.
        eta_EI : float
            Learning rate for excitatory-inhibitory connections.
        eta_IE : float
            Learning rate for inhibitory-excitatory connections.
        eta_II : float
            Learning rate for inhibitory-inhibitory connections.
        rho_E : float
            Target firing rate for excitatory neurons.
        rho_I : float
            Target firing rate for inhibitory neurons.
        """
        self.eta_EE = eta_EE
        self.eta_II = eta_II
        self.eta_EI = eta_EI
        self.eta_IE = eta_IE
        
        self.rho_E = rho_E
        self.rho_I = rho_I
        
        self.plasticity = True
        
        self.params_plast = (self.eta_EE, self.eta_EI, self.eta_IE, self.eta_II, self.rho_E, self.rho_I)
        
    def run_plasticity(self, u):
        """
        Wrapper to run the plasticity step.

        Parameters
        ----------
        u : array
            Input to the network.
        """

        if not self.plasticity:
            raise utils.NotInitialized("Plasticity parameters have not been initialized yet")
            
        args = (u, self.E0, self.I0,
                *self.params_plast, *self.params_dyn)
                        
        results = utils.plasticity_numba(*args)

        self.E_plasticity, self.I_plasticity = results[:2]
        self.WEE, self.WEI, self.WIE, self.WII = results[2:]
        
    def run_recurrent(self, u, E0 = None, I0 = None):
        """
        Wrapper to integrate the dynamics of the network.

        Parameters
        ----------
        u : array
            Input to the network.
        E0 : array, optional
            Initial condition for excitatory neurons.
            The default is None, in which case the default initial condition is used.
        I0 : array, optional
            Initial condition for inhibitory neurons.
            The default is None, in which case the default initial condition is used.

        Returns
        -------
        results : tuple
            Tuple containing the results of the integration.
        """

        assert u.shape[1] == self.NInputs
        
        if E0 is None:
            E0 = self.E0
        if I0 is None:
            I0 = self.I0
        
        results = utils.integrate_dynamics(u, E0, I0, *self.params_dyn)

        return results
    
    def train_model(self, u, initTraining, alpha, max_iter = 1000,
                    plot = True, set_Wout = True):
        """
        Train the model with linear regression.

        Parameters
        ----------
        u : array
            Input to the network.
        initTraining : int
            Initial time step for training.
        alpha : float
            Regularization parameter for linear regression.
        max_iter : int, optional
            Maximum number of iterations for linear regression.
            The default is 1000.
        plot : bool, optional
            Whether to plot the results or not.
            The default is True.
        set_Wout : bool, optional
            Whether to set the output weights or not.
            The default is True.

        Returns
        -------
        Wout : array
            Output weights of the trained model.
            Only returned if set_Wout is True.
        MSE : float
            Mean squared error of the trained model.
        trained_output : array
            Output of the trained model.
        """
        if self.generate_recurrent == True:
            self.E_rec, self.I_rec = self.run_recurrent(u)
            self.generate_recurrent = False

        self.target = u[initTraining + 1:].T
        self.inputs = u[initTraining:-1]
        X = self.E_rec[initTraining + 1:, :]
        
        regression = linear_model.Ridge(alpha = alpha, max_iter = max_iter, fit_intercept = False)
        regression_fit = regression.fit(X, self.target.T)
        Wout = regression_fit.coef_

        trained_output = np.dot(Wout, X.T)
        MSE = np.sqrt(np.mean((trained_output - self.target)**2, axis = 1)).mean()

        if plot:
            fig, axs = plt.subplots(figsize = (20,10), nrows = 3)
            for i in range(3):
                axs[i].plot(trained_output[i], c = 'orangered', lw = 2, label = 'Trained')
                axs[i].plot(self.target[i], ls = '--', c = 'navy', lw = 1, label = 'Target')
            axs[2].set_xlabel('Timestep', fontsize = 20)
            axs[0].set_ylabel('$x$', fontsize = 20)
            axs[1].set_ylabel('$y$', fontsize = 20)
            axs[2].set_ylabel('$z$', fontsize = 20)
        
            plt.show()
        
        if set_Wout:
            self.Wout = Wout
            return MSE, trained_output
        else:
            return Wout, MSE, trained_output
    
    def echo_state(self, u, nLoops, idx_start, idx_echo, plot = False, Wout = None):
        """
        Wrapper to run the echo state.

        Parameters
        ----------
        u : array
            Input to the network.
        nLoops : int
            Number of steps in the echo state to run.
        idx_start : int
            Initial time step for the echo state.
        idx_echo : int
            Number of steps to run before the echo state.
        plot : bool, optional
            Whether to plot the results or not.
            The default is False.
        Wout : array, optional
            Output weights of the network.
            The default is None, in which case the default output weights are used.
        """
        E0 = self.E_rec[idx_start]
        I0 = self.I_rec[idx_start]

        if Wout is None:
            Wout = self.Wout

        E_echo, I_echo, output_echo = utils.run_echo_state(nLoops, idx_start, idx_echo, Wout,
                                                           u, E0, I0, *self.params_dyn)

        if plot:
            Steps = np.arange(idx_start, idx_start + idx_echo + nLoops + 1)

            fig, axs = plt.subplots(figsize = (20,10), nrows = 4)
            for i in range(3):
                axs[i].plot(Steps, output_echo[:, i], c = 'orangered', lw = 2)
                axs[i].plot(np.arange(u.shape[0]), u[:, i], ls = '--', c = 'navy', lw = 1)
                axs[i].axvline(idx_start + idx_echo, c = 'darkred')
                axs[i].set_xlim(idx_start, idx_start + idx_echo + nLoops + 1)
            axs[0].set_ylabel('$x$', fontsize = 20)
            axs[1].set_ylabel('$y$', fontsize = 20)
            axs[2].set_ylabel('$z$', fontsize = 20)
            axs[3].set_xlabel('Timestep', fontsize = 20)
            axs[3].set_ylabel('E nodes', fontsize = 20)
                
            axs[-1].pcolormesh(Steps, np.arange(self.NE), E_echo.T,
                               cmap = "turbo", shading = 'auto',
                               vmin = 0, vmax = 1)
            cb_ax = fig.add_axes([.88,.12,.04,.15])
            cb_ax.axis('off')
            cbar = fig.colorbar(axs[-1].collections[0], ax = cb_ax)
            cbar.set_label('$E$', fontsize = 20, rotation = 270, labelpad = 20)

            plt.show()

        return E_echo, I_echo, output_echo
            
    def predict(self, E):
        """
        Predict the output of the network given the state of the E nodes.

        Parameters
        ----------
        E : array
            State of the E nodes.

        Returns
        -------
        output : array
            Output of the network.
        """
        return np.dot(self.Wout, E)
    
    def return_input(self, ut):
        """
        Return the input to the network trough the input weights.

        Parameters
        ----------
        ut : array
            Input to the network at time t.

        Returns
        -------
        array
            Input to the network.
        """     
        return np.dot(ut, self.Win)
    
    def dynamical_step(self, E, I, ut):
        """
        Helper method to perform a dynamical step.

        Parameters
        ----------
        E : array
            State of the E nodes at time t.
        I : array
            State of the I nodes at time t.
        ut : array
            Input to the network at time t.

        Returns
        -------
        new_E : array
            State of the E nodes at time t + dt.
        new_I : array
            State of the I nodes at time t + dt.
        """
        new_E, new_I = self.dynamical_step(E, I, ut, *self.params_dyn)
        
        return new_E, new_I
    
    def plot_attractor(self, u):
        """
        Plot the attractor corresponding to the trajectory u.

        Parameters
        ----------
        u : array
            Trajectory.
        """
        fig = plt.figure(figsize = (20,5))
        axs = fig.subplot_mosaic('A03;A14;A25')
        ss = axs['A'].get_subplotspec()
        axs['A'].remove()
        axs['A'] = fig.add_subplot(ss, projection='3d')
        axs['A'].plot(*u.T, alpha=0.7, linewidth=1)

        for i in range(3):
            axs[str(i)].plot(u[:,i])
            axs[str(i)].axis('off')
            axs[str(i + 3)].scatter(*utils.return_map(u[:,i]), s = 10)
            axs[str(i + 3)].axis('off')
        plt.show()

    def compute_histograms(self, vals, bins):
        """
        Compute the histograms of the values in vals.

        Parameters
        ----------
        vals : array    
            Values to compute the histograms of.
        bins : array
            Bins to use for the histograms.

        Returns
        -------
        h : array
            Histograms of the values in vals.
        """
        h = np.zeros((3, bins.size-1), dtype=np.float64)

        for i in range(3):
            h[i] = np.histogram(vals[:,i], bins = bins, density = True)[0]

        return h
    
    def histogram_distance(self, h1, h2):
        """
        Compute the distance between two histograms using the same bins.

        Parameters
        ----------
        h1 : array
            First histogram.
        h2 : array
            Second histogram.
        
        Returns
        -------
        float
            Distance between the two histograms.
        """

        return np.sqrt(np.sum((h1 - h2)**2, axis = 1)).mean()
    
    def compare_results(self, u, output_echo, NBins, plot = True):
        """
        Compare the results of the reservoir with the real trajectory.

        Parameters
        ----------
        u : array
            Real trajectory.
        output_echo : array
            Output of the reservoir.
        NBins : int
            Number of bins to use for the histograms.
        plot : bool, optional
            Whether to plot the results. The default is True.

        Returns
        -------
        float
            Distance between the histograms of the real trajectory and the output of the reservoir.
        """
        max_val = np.max(u)
        max_val = np.max([max_val, np.max(output_echo)])

        min_val = np.min(u)
        min_val = np.min([min_val, np.min(output_echo)])

        bins = np.linspace(min_val, max_val, NBins)

        h_real = self.compute_histograms(u, bins = bins)
        h_echo = self.compute_histograms(output_echo, bins = bins)

        if plot:
            fig, axs = plt.subplots(figsize = (20,5), ncols = 3)
            for i in range(3):
                axs[i].bar((bins[1:] + bins[:-1])/2, h_real[i], width = np.diff(bins),
                           color = 'navy', label = 'Real', zorder = 0)
                axs[i].bar((bins[1:] + bins[:-1])/2, h_echo[i], width = np.diff(bins),
                           color = 'orangered', alpha = 0.5, label = 'Echo', zorder = 1)
                axs[i].legend(fontsize = 20)
            axs[2].set_xlabel('Timestep', fontsize = 20)
            axs[0].set_ylabel('$x$', fontsize = 20)
            axs[1].set_ylabel('$y$', fontsize = 20)
            axs[2].set_ylabel('$z$', fontsize = 20)

            plt.subplots_adjust(wspace=0.3)
        
            plt.show()

        return self.histogram_distance(h_real, h_echo)

    def fit_alpha(self, u, initTraining, idx_start, idx_echo, nLoops,
                  alpha_min = 1e-6, alpha_max = 1e-2, NAlpha = 10, NBins = 100):
        """
        Fit the optimal regularization parameter alpha.

        Parameters
        ----------
        u : array
            Input to the reservoir.
        initTraining : int
            Initial time step for training.
        idx_start : int
            Initial time step for the echo state phase (with clamping).
        idx_echo : int
            Time step at which the echo state phase starts, without clamping.
        nLoops : int
            Number of loops to perform in the echo state phase.
        alpha_min : float, optional
            Minimum value of alpha to try. The default is 1e-6.
        alpha_max : float, optional
            Maximum value of alpha to try. The default is 1e-2.
        NAlpha : int, optional
            Number of values of alpha to try. The default is 10.
        NBins : int, optional
            Number of bins to use for the histograms. The default is 100.
        """
        self.trained = False

        alpha_array = np.linspace(alpha_min, alpha_max, NAlpha)
        self.MSE_echo = np.inf

        for alpha in alpha_array:
            Wout, _, trained_output = self.train_model(u, initTraining, alpha = alpha,
                                                       plot = False, set_Wout = False)
            E_echo, I_echo, output_echo = self.echo_state(u, nLoops, idx_start = idx_start,
                                                          idx_echo = idx_echo, plot = False,
                                                          Wout = Wout)

            MSE_echo = self.compare_results(u, output_echo[idx_echo:], NBins, plot = False)

            if MSE_echo < self.MSE_echo:
                    self.MSE_echo = MSE_echo
                    self.Wout = Wout
                    self.trained_output = trained_output
                    self.alpha = alpha
                    self.E_echo = E_echo
                    self.I_echo = I_echo
                    self.output_echo = output_echo

                    print('New MSE:', np.round(self.MSE_echo, 5), 'for alpha =', self.alpha)

        self.trained = True