import numpy as np
from scipy.optimize import root, minimize, shgo, dual_annealing, newton, basinhopping, least_squares
from load_derivs import load1, load2, derivs
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
#constants in cgs
M_SOL = 1.989e33
G = 6.67430e-8
a= 7.5657e-15
c= 2.99792458e10
sigma = a*c/4
N_A = 6.02214076e23
k_B = 1.380649e-16
#initial guesses for star - scaled roughly from solar values
P_c_init = 10**17
T_c_init = 10**7
L_star_init = 2*10**33
R_star_init = 5*10**10
#adjustable parametrs: mass and composition
M = 0.9*M_SOL
X = 0.70
Y = 0.28
Z = 0.02
# calculate ionization mean molecular weight
mu_ion = 1 / (X + Y/4 + Z/14)
# calculate electron mean molecular weight
mu_e = 1 / (X + Y/2 + Z/2)
# calculate total mean molecular weight
mu = 1 / (1/mu_ion + 1/mu_e)


#set up mass spans
m_span_1 = [M*1e-3, M*0.5]
m_span_2 = [M*1.0, M*0.5]
m_eval_1 = np.linspace(m_span_1[0], m_span_1[1], 1000)
m_eval_2 = np.linspace(m_span_2[0], m_span_2[1], 1000)


#least sqares optimization function using scaled parameters for balaned convergence
def func(scales):
    try:
        print(f"Evaluating scales: {scales}")
        #convert scales back to actual values
        P_c = P_c_init * np.abs(scales[0])
        T_c = T_c_init * np.abs(scales[1])
        L_star = L_star_init * np.abs(scales[2])
        R_star = R_star_init * np.abs(scales[3])
        #load initial conditions
        initial_1 = load1(m_span_1[0],T_c,P_c)
        initial_2 = load2(m_span_2[0],L_star,R_star)
        #shoot from both ends and calculate difference at midpoint
        sol_1 = solve_ivp(derivs,m_span_1,initial_1,t_eval=m_eval_1,method='RK45')
        sol_2 = solve_ivp(derivs,m_span_2,initial_2,t_eval=m_eval_2,method='RK45')
        L_diff = sol_1.y[0][-1] - sol_2.y[0][-1]
        P_diff = sol_1.y[1][-1] - sol_2.y[1][-1]
        R_diff = sol_1.y[2][-1] - sol_2.y[2][-1]
        T_diff = sol_1.y[3][-1] - sol_2.y[3][-1]
        
        #normalized error components
        arr = [P_diff/P_c_init, T_diff/T_c_init, L_diff/L_star_init, R_diff/R_star_init]
        #minimize RMS error
        rms_error = np.sqrt(np.sum(np.array(arr)**2))
        print(f"Residuals: {arr}, RMS Error: {rms_error:.6e}")
        return rms_error
    #error handling when optimiation algorithm tries unphysical parameters
    except (ValueError,IndexError) as e:
        print(f"ValueError encountered: {e}")
        #return large error to discourage this parameter set
        return np.inf



#initial scale and bounded region
x0 = [1.0, 1.0, 1.0, 1.0]
#tighter bounds on luminosity and radius given known stellar properties
bounds = [(0.5,5.0), (0.5,5.0), (0.5,2.0), (0.7,2.0)]
#use global optimization algorithm to avoid getting stuck in local minima
global_result = dual_annealing(func, bounds, maxiter=50, x0=x0,no_local_search=True)

#output result and converged values
print(global_result)
converged_vals_new = [P_c_init*np.abs(global_result.x[0]), T_c_init*np.abs(global_result.x[1]), L_star_init*np.abs(global_result.x[2]), R_star_init*np.abs(global_result.x[3])]
print(converged_vals_new)

#load converged run
initial_1 = load1(m_span_1[0],converged_vals_new[1],converged_vals_new[0])
initial_2 = load2(m_span_2[0],converged_vals_new[2],converged_vals_new[3])

sol_1 = solve_ivp(derivs,m_span_1,initial_1,t_eval=m_eval_1,method='RK45')
sol_2 = solve_ivp(derivs,m_span_2,initial_2,t_eval=m_eval_2,method='RK45')



#save results to dataframe
df_1 = pd.DataFrame({'Mass (g)': sol_1.t, 'Luminosity (erg/s)': sol_1.y[0], 'Pressure (dyn/cm^2)': sol_1.y[1], 'Radius (cm)': sol_1.y[2], 'Temperature (K)': sol_1.y[3]})
df_2 = pd.DataFrame({'Mass (g)': sol_2.t, 'Luminosity (erg/s)': sol_2.y[0], 'Pressure (dyn/cm^2)': sol_2.y[1], 'Radius (cm)': sol_2.y[2], 'Temperature (K)': sol_2.y[3]})
#df_1.to_csv('../shoot_solution_1.csv', index=False)
#df_2.to_csv('../shoot_solution_2.csv', index=False)


