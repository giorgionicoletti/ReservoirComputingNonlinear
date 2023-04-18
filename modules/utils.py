import numpy as np
from numba import njit, prange

import scipy.signal

class NotInitialized(Exception):
    """
    A custom exception class, raised when a variable is not initialized.

    Parameters
    ----------
    err : str
        The error message to be printed.

    Attributes
    ----------
    err : str
        The error message to be printed.
    
    Methods
    -------
    __init__(self, err = "Not itialized")
        The constructor.
    """
    def __init__(self, err = "Not itialized"):
        self.err = err
        super().__init__(self.err)


########################
# Activation functions #
########################

@njit(fastmath=True)
def ReLU(x):
    """
    Fast implementation of the ReLU function.

    Parameters
    ----------
    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The output array passed through the ReLU function.
    """
    array = np.zeros(x.shape, dtype = np.float64)
    return np.maximum(array, x)

@njit(fastmath=True)
def ReLU_Sat(x, sat = 10):
    """
    Implementation of the ReLU function with saturation.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    sat : float
        The saturation value.

    Returns
    -------
    np.ndarray
        The output array passed through the ReLU function with saturation.
    """

    x = (x > 0) * x
    return np.minimum(x, sat)

@njit(fastmath=True)
def ReLU_Poly(x, eps = 0):
    """
    Implementation of the ReLU function with second-order polynomial correction.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    eps : float
        The strength of the polynomial correction.

    Returns
    -------
    np.ndarray
        The output array passed through the ReLU function with second-order polynomial correction.
    """
    return (x > 0) * (x + eps*x**2)

@njit(fastmath=True)
def ReLU_Log(x, eps = 0):
    """
    Implementation of the ReLU function with logarithmic correction.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    eps : float
        The strength of the logarithmic correction.

    Returns
    -------
    np.ndarray
        The output array passed through the ReLU function with logarithmic correction.
    """
    return (x > 0) * (x + eps*np.log(x + 1))

@njit(fastmath=True)
def sigmoid(x):
    """
    Fast implementation of the sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        The output array passed through the sigmoid function.
    """
    return 1/(1 + np.exp(-x))

####################
# Lorenz functions #
####################

def lorenz_derivative(t, xyz, sigma, rho, beta):
    """
    The Lorenz system of differential equations.

    Parameters
    ----------
    t : float
        The time.
    xyz : np.ndarray
        The state vector.
    sigma : float
        The sigma parameter.
    rho : float
        The rho parameter.
    beta : float
        The beta parameter.

    Returns
    -------
    np.ndarray
        The derivative of the state vector.
    """

    x, y, z = xyz
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    return [dx_dt, dy_dt, dz_dt]

def return_map(data):
    """
    Returns the return map of a given time series.

    Parameters
    ----------
    data : np.ndarray
        The time series.

    Returns
    -------
    np.ndarray
        The return map.
    """

    M = scipy.signal.argrelextrema(data, np.greater)[0]
    return data[M[:-1]], data[M[1:]]


####################
# Dynamical system #
####################

@njit(fastmath = True)
def dotE_dotI(E, I, ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args):
    """
    The derivative of the state of the dynamical system.

    Parameters
    ----------
    E : np.ndarray
        The state of the excitatory population.
    I : np.ndarray
        The state of the inhibitory population.
    ut : np.ndarray
        The input vector at time t.
    tau_E : float
        The characteristic time of the excitatory population.
    tau_I : float
        The characteristic time of the inhibitory population.
    Win : np.ndarray
        The input weights.
    WEE : np.ndarray
        The excitatory to excitatory weights.
    WEI : np.ndarray
        The inhibitory to excitatory weights.
    WIE : np.ndarray
        The excitatory to inhibitory weights.
    WII : np.ndarray
        The inhibitory to inhibitory weights.
    biases_E : np.ndarray
        The biases of the excitatory population.
    biases_I : np.ndarray
        The biases of the inhibitory population.
    gamma : float
        The strength of the recurrent connections.
    fun_nl : function
        The nonlinearity function.
    args : tuple
        The arguments of the nonlinearity function.
    
    Returns
    -------
    dotE : np.ndarray
        The derivative of the state of the excitatory population.
    dotI : np.ndarray
        The derivative of the state of the inhibitory population.
    """

    dotE = 1/tau_E*(-E + gamma*fun_nl(np.dot(WEE, E) - np.dot(WEI, I) + np.dot(ut, Win) - biases_E, *args))
    dotI = 1/tau_I*(-I + fun_nl(np.dot(WIE, E) - np.dot(WII, I) - biases_I, *args))

    return dotE, dotI

@njit(fastmath = True)
def rk4_step(E, I, ut, dt, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args):
    """
    The Runge-Kutta 4th order integration step.

    Parameters
    ----------
    E : np.ndarray
        The state of the excitatory population.
    I : np.ndarray
        The state of the inhibitory population.
    ut : np.ndarray
        The input vector at time t.
    tau_E : float
        The characteristic time of the excitatory population.
    tau_I : float
        The characteristic time of the inhibitory population.
    Win : np.ndarray
        The input weights.
    WEE : np.ndarray
        The excitatory to excitatory weights.
    WEI : np.ndarray
        The inhibitory to excitatory weights.
    WIE : np.ndarray
        The excitatory to inhibitory weights.
    WII : np.ndarray
        The inhibitory to inhibitory weights.
    biases_E : np.ndarray
        The biases of the excitatory population.
    biases_I : np.ndarray
        The biases of the inhibitory population.
    gamma : float
        The strength of the recurrent connections.
    fun_nl : function
        The nonlinearity function.
    args : tuple
        The arguments of the nonlinearity function.
    
    Returns
    -------
    dotE : np.ndarray
        The derivative of the state of the excitatory population.
    dotI : np.ndarray
        The derivative of the state of the inhibitory population.
    """
    f1_E, f1_I = dotE_dotI(E, I, ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, *args)
    f2_E, f2_I = dotE_dotI(E + f1_E*dt/2, I + f1_I*dt/2,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, *args)
    f3_E, f3_I = dotE_dotI(E + f2_E*dt/2, I + f2_I*dt/2,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, *args)
    f4_E, f4_I = dotE_dotI(E + f3_E*dt, I + f3_I*dt,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, *args)

    return E + dt/6*(f1_E + 2*f2_E + 2*f3_E + f4_E), I + dt/6*(f1_I + 2*f2_I + 2*f3_I + f4_I)


@njit
def dynamical_step(ut, E, I, dt_E, dt_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args):
    """
    Performs a single step of the dynamical system.

    Parameters
    ----------
    ut : np.ndarray
        The input vector.
    E : np.ndarray
        The state of the excitatory population.
    I : np.ndarray
        The state of the inhibitory population.
    dt_E : float
        The time step normalized with the characteristic time of
        the excitatory population.
    dt_I : float
        The time step normalized with the characteristic time of
        the inhibitory population.
    Win : np.ndarray
        The input weights.
    WEE : np.ndarray
        The excitatory to excitatory weights.
    WEI : np.ndarray
        The inhibitory to excitatory weights.
    WIE : np.ndarray
        The excitatory to inhibitory weights.
    WII : np.ndarray
        The inhibitory to inhibitory weights.
    biases_E : np.ndarray
        The biases of the excitatory population.
    biases_I : np.ndarray
        The biases of the inhibitory population.
    gamma : float
        The gain of the excitatory population.
    fun_nl : function
        The nonlinearity function.
    args : tuple
        The arguments of the nonlinearity function.
    
    Returns
    -------
    np.ndarray
        The new state of the excitatory population.
    np.ndarray
        The new state of the inhibitory population.
    """
        
    new_E = E + dt_E*(-E + gamma*fun_nl(np.dot(WEE, E) - np.dot(WEI, I) + np.dot(ut, Win) - biases_E, *args))
    new_I = I + dt_I*(-I + fun_nl(np.dot(WIE, E) - np.dot(WII, I) - biases_I, *args))

    return new_E, new_I

@njit(nogil = True)
def recurrent_numba(u, E0, I0,
                    dt, Win, WEE, WEI, WIE, WII, tau_E, tau_I, biases_E, biases_I, gamma, fun_nl = ReLU):
    """
    Performs the simulation of the dynamical system.

    Parameters
    ----------
    u : np.ndarray
        The input time series.
    E0 : np.ndarray
        The initial state of the excitatory population.
    I0 : np.ndarray
        The initial state of the inhibitory population.
    dt : float
        The time step.
    Win : np.ndarray
        The input weights.
    WEE : np.ndarray
        The excitatory to excitatory weights.
    WEI : np.ndarray
        The inhibitory to excitatory weights.
    WIE : np.ndarray
        The excitatory to inhibitory weights.
    WII : np.ndarray
        The inhibitory to inhibitory weights.
    tau_E : float
        The characteristic time of the excitatory population.
    tau_I : float
        The characteristic time of the inhibitory population.
    biases_E : np.ndarray
        The biases of the excitatory population.
    biases_I : np.ndarray
        The biases of the inhibitory population.
    gamma : float
        The gain of the excitatory population.

    Returns
    -------
    np.ndarray
        The state of the excitatory population at all times.
    np.ndarray
        The state of the inhibitory population at all times.
    """
    NE, NI = WEI.shape
    nSteps = u.shape[0]
    
    E = np.zeros((nSteps, NE), dtype = np.float64)
    I = np.zeros((nSteps, NI), dtype = np.float64)

    E[0] = E0
    I[0] = I0
    
    dt_E = dt/tau_E
    dt_I = dt/tau_I
    
    for t in range(nSteps - 1):
        E[t+1], I[t+1] = dynamical_step(u[t], E[t], I[t], dt_E, dt_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl)
                
    return E, I


@njit(parallel = True)
def plasticity_numba(u, E0, I0,
                     eta_EE, eta_EI, eta_IE, eta_II, rho_E, rho_I,
                     dt, Win, WEE, WEI, WIE, WII, tau_E, tau_I, biases_E, biases_I, gamma, fun_nl = ReLU):
    """
    Performs the simulation of the dynamical system with plasticity.

    Parameters
    ----------
    u : np.ndarray
        The input time series.
    E0 : np.ndarray
        The initial state of the excitatory population.
    I0 : np.ndarray
        The initial state of the inhibitory population.
    eta_EE : float
        The learning rate of the excitatory to excitatory weights.
    eta_EI : float
        The learning rate of the inhibitory to excitatory weights.
    eta_IE : float
        The learning rate of the excitatory to inhibitory weights.
    eta_II : float
        The learning rate of the inhibitory to inhibitory weights.
    rho_E : float
        The target rate of the excitatory population.
    rho_I : float
        The target rate of the inhibitory population.
    dt : float
        The time step.
    Win : np.ndarray
        The input weights.
    WEE : np.ndarray
        The excitatory to excitatory weights.
    WEI : np.ndarray
        The inhibitory to excitatory weights.
    WIE : np.ndarray
        The excitatory to inhibitory weights.
    WII : np.ndarray
        The inhibitory to inhibitory weights.
    tau_E : float
        The characteristic time of the excitatory population.
    tau_I : float
        The characteristic time of the inhibitory population.
    biases_E : np.ndarray
        The biases of the excitatory population.
    biases_I : np.ndarray
        The biases of the inhibitory population.
    gamma : float
        The gain of the excitatory population.

    Returns
    -------
    np.ndarray
        The state of the excitatory population at all times.
    np.ndarray
        The state of the inhibitory population at all times.
    np.ndarray
        The final excitatory to excitatory weights.
    np.ndarray
        The final inhibitory to excitatory weights.
    np.ndarray
        The final excitatory to inhibitory weights.
    np.ndarray
        The final inhibitory to inhibitory weights.
    """

    NE, NI = WEI.shape
    nSteps = u.shape[0]
    
    E = np.zeros((nSteps, NE), dtype = np.float64)
    I = np.zeros((nSteps, NI), dtype = np.float64)

    E[0] = E0
    I[0] = I0
    
    sum_from_input = np.sum(Win, axis = 0)
    
    dt_E = dt/tau_E
    dt_I = dt/tau_I
    
    for t in range(nSteps - 1):
        Input = np.dot(u[t], Win)

        E[t+1], I[t+1] = dynamical_step(u[t], E[t], I[t], dt_E, dt_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl)
                        
        for i in prange(NE):            
            SumIncoming = 0.
            for j in range(NE):
                if j != i:
                    DeltaWEE = E[t+1, i]*E[t+1, j]
                    WEE[i, j] += eta_EE*DeltaWEE

                    if WEE[i, j] < 0:
                        WEE[i, j] = 0.
                    else:
                        SumIncoming += WEE[i, j]
            WEE[i] /= (SumIncoming + sum_from_input[i])
            
            SumIncoming = 0.
            for j in range(NI):
                DeltaWEI = I[t+1,j]*(E[t+1, i] - rho_E)
                WEI[i, j] += eta_EI*DeltaWEI
                if WEI[i, j] < 0:
                    WEI[i, j] = 0.
                else:
                    SumIncoming += WEI[i, j]        
            WEI[i] /= (SumIncoming + sum_from_input[i])
        
        for i in prange(NI):            
            SumIncoming = 0.

            for j in range(NI):
                if j != i:
                    DeltaWII = I[t+1, j]*(I[t+1, i] - rho_I)
                    WII[i, j] += eta_II*DeltaWII

                    if WII[i, j] < 0:
                        WII[i, j] = 0.
                    else:
                        SumIncoming += WII[i, j]
            WII[i] /= SumIncoming
            
            SumIncoming = 0.
            for j in range(NE):
                DeltaWIE = I[t+1,i]*E[t+1, j]
                WIE[i, j] += eta_IE*DeltaWIE
                if WIE[i, j] < 0:
                    WIE[i, j] = 0.
                else:
                    SumIncoming += WIE[i, j]
            WIE[i] /= SumIncoming
                    
    return E, I
