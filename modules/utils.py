import numpy as np
from numba import njit, prange

import scipy.signal
from concurrent.futures import ProcessPoolExecutor

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

class NotImplemented(Exception):
    """
    A custom exception class, raised when something is not implemented.

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
    def __init__(self, err = "Not implemented"):
        self.err = err
        super().__init__(self.err)


def parallelize(func, iterable, global_variables, num_cores):
    """
    Parallelize a function using the concurrent.futures module.
    RAM-consuming arguments should be passed as global variables.

    Parameters
    ----------
    func : function
        function to be parallelized
    iterable : list
        list of arguments to be passed to the function.
        These arguments will be copied to each process, so it is best to pass a list of small objects
        and declare shared variables in the global namespace, using the global_variables parameter
    global_variables : dict
        dictionary of global variables to be passed to the function
        This prevents copies of the arguments that may result in large memory usage
        Global variables are created in the scope of the function, and the keys of the dictionary are the
        names of the variables used in func. The values of the dictionary are the values of the variables.
    num_cores : int
        number of cores to use
    
    Returns
    -------
    result : iterator
        iterator of the results
    """
    # iterate through the dictionary global_variables and add the variables to the global namespace
    for key, value in global_variables.items():
        globals()[key] = value

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        result = executor.map(func, iterable)
        
    return result



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
def ReLU_Poly(x, eps = 0, sat = 10, exp = 2):
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
    x = (x > 0) * x
    return np.minimum(x + eps*x**exp, sat)

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

    x = (x > 0) * x

    return x + eps*np.log(x + 1)

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
    f1_E, f1_I = dotE_dotI(E, I,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)
    f2_E, f2_I = dotE_dotI(E + f1_E*dt/2, I + f1_I*dt/2,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)
    f3_E, f3_I = dotE_dotI(E + f2_E*dt/2, I + f2_I*dt/2,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)
    f4_E, f4_I = dotE_dotI(E + f3_E*dt, I + f3_I*dt,
                           ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)

    return E + dt/6*(f1_E + 2*f2_E + 2*f3_E + f4_E), I + dt/6*(f1_I + 2*f2_I + 2*f3_I + f4_I)

@njit(fastmath = True)
def euler_step(E, I, ut, dt, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args):
    """
    The Euler integration step.

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
    f1_E, f1_I = dotE_dotI(E, I, ut, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)

    return E + dt*f1_E, I + dt*f1_I

@njit
def integrate_dynamics(u, E0, I0,
                       dt, tau_E, tau_I,
                       Win, WEE, WEI, WIE, WII,
                       biases_E, biases_I, gamma, fun_nl, args, dynamical_step):
    """
    Integrate dynamics using a custom dynamical step, such as RK4 or a Euler step.

    Parameters
    ----------
    u : np.ndarray
        The input vector.
    E0 : np.ndarray
        The initial state of the excitatory population.
    I0 : np.ndarray
        The initial state of the inhibitory population.
    dt : float
        The time step.
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
    dynamical_step : function
        The function that performs the dynamical step. It must be a function
        decorated with @njit, and compatible with the arguments of this function.
        Passing the function directly allows for higher level compilation in the
        class that wraps the numba-enabled functions.

    Returns
    -------
    E : np.ndarray
        The state of the excitatory population.
    I : np.ndarray
        The state of the inhibitory population.
    """

    NE, NI = WEI.shape
    nSteps = u.shape[0]
    
    E = np.zeros((nSteps, NE), dtype = np.float64)
    I = np.zeros((nSteps, NI), dtype = np.float64)

    E[0] = E0
    I[0] = I0
        
    for t in range(nSteps - 1):
        E[t+1], I[t+1] = dynamical_step(E[t], I[t], u[t], dt, tau_E, tau_I, Win, WEE, WEI, WIE, WII, biases_E, biases_I, gamma, fun_nl, args)
                
    return E, I

@njit
def run_echo_state(nLoops, idx_start, idx_echo, Wout,
                   u, E0, I0,
                   dt, tau_E, tau_I,
                   Win, WEE, WEI, WIE, WII,
                   biases_E, biases_I, gamma, fun_nl, args, dynamical_step):
    """
    Run the echo state dynamics by clamping the input for a number of steps first,
    and then feeding the output to the input layer and letting the network evolve.

    Parameters
    ----------
    nLoops : int
        The number of steps to run the network during the echo state phase.
    idx_start : int
        The index of the first step of the clamped input phase.
    idx_echo : int
        The index of the first step of the echo state phase.
    Wout : np.ndarray
        The output weights.
    u : np.ndarray
        The input vector.
    E0 : np.ndarray
        The initial state of the excitatory population.
    I0 : np.ndarray
        The initial state of the inhibitory population.
    dt : float
        The time step.
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
    dynamical_step : function
        The function that performs the dynamical step. It must be a function
        decorated with @njit, and compatible with the arguments of this function.
        Passing the function directly allows for higher level compilation in the
        class that wraps the numba-enabled functions.

    
    Returns
    -------
    E : np.ndarray
        The state of the excitatory population.
    I : np.ndarray
        The state of the inhibitory population.
    output_sequence : np.ndarray
        The output sequence during the clamp phase and the echo state phase.
    """

    NE, NI = WEI.shape
    _, NInputs = u.shape
    
    E = np.zeros((nLoops + 1 + idx_echo, NE), dtype = np.float64)
    I = np.zeros((nLoops + 1 + idx_echo, NI), dtype = np.float64)

    E[0] = E0
    I[0] = I0

    output_sequence = np.zeros((nLoops + 1 + idx_echo, NInputs), dtype = np.float64)

    output_sequence[0] = np.dot(Wout, E[0])

    for t in range(idx_echo):
        E[t+1], I[t+1] = dynamical_step(E[t], I[t], u[idx_start + t], dt, tau_E, tau_I,
                                        Win, WEE, WEI, WIE, WII,
                                        biases_E, biases_I, gamma, fun_nl, args)
        output_sequence[t+1] = np.dot(Wout, E[t+1])

    for t in range(idx_echo, nLoops + idx_echo):
        E[t+1], I[t+1] = dynamical_step(E[t], I[t], output_sequence[t], dt, tau_E, tau_I,
                                        Win, WEE, WEI, WIE, WII,
                                        biases_E, biases_I, gamma, fun_nl, args)
        output_sequence[t+1] = np.dot(Wout, E[t+1])

    return E, I, output_sequence


@njit(parallel = True)
def plasticity_numba(u, E0, I0,
                     eta_EE, eta_EI, eta_IE, eta_II, rho_E, rho_I,
                     dt, tau_E, tau_I, Win, WEE, WEI, WIE, WII,
                     biases_E, biases_I, gamma, fun_nl, args, dynamical_step):
    """
    Performs the simulation of the dynamical system with plasticity, by changing
    the weights during the dynamics of the network with the external input.

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
    fun_nl : function
        The nonlinearity function.
    args : tuple
        The arguments of the nonlinearity function.
    dynamical_step : function
        The function that performs the dynamical step. It must be a function
        decorated with @njit, and compatible with the arguments of this function.
        Passing the function directly allows for higher level compilation in the
        class that wraps the numba-enabled functions.

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
    
    sum_from_input = np.zeros(NE, dtype = np.float64)
    for i in range(NE):
        sum_from_input[i] = np.sum(Win[:,i])

        
    for t in range(nSteps - 1):
        Input = np.dot(u[t], Win)

        E[t+1], I[t+1] = dynamical_step(E[t], I[t], u[t], dt, tau_E, tau_I,
                                        Win, WEE, WEI, WIE, WII, biases_E, biases_I,
                                        gamma, fun_nl, args)
        
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
                    
    return E, I, WEE, WEI, WIE, WII
