import numpy as np
from calculate_density import calculate_density
from energy_generation import energy_generation
from opacity import opacity
from scipy.optimize import fsolve

#calculate_density(0, 0.98, 0.02, 10**7.55, 10**16.85)[0]
#print(energy_generation(10**(7.25), 10**(1.90), 0.70, 0.28, 0.02))
#constants in cgs
M_SOL = 1.989e33
G = 6.67430e-8
a= 7.5657e-15
c= 2.99792458e10
sigma = a*c/4
N_A = 6.02214076e23
k_B = 1.380649e-16
#adjustable parametrers: composition
X = 0.70
Y = 0.28
Z = 0.02
# calculate ionization mean molecular weight
mu_ion = 1 / (X + Y/4 + Z/14)
# calculate electron mean molecular weight
mu_e = 1 / (X + Y/2 + Z/2)
# calculate total mean molecular weight
mu = 1 / (1/mu_ion + 1/mu_e)

def load1(m, T_c, P_c):
    #calculate central density
    rho_c = calculate_density(X,Y,Z,T_c,P_c)[0]
    #calculate ratio of gas pressure to total pressure
    beta = calculate_density(X,Y,Z,T_c,P_c)[1]
    #calculate energy generation rate
    eps = energy_generation(T_c, rho_c, X, Y, Z)
    #get opacity from table
    kappa = opacity(rho_c, T_c)

    #determine nabla_ad based on beta
    nabla_ad = 2*(4- 3*beta)/(32-24*beta-3*beta**2)
    #nabla_ad = 0.4
    #determine nabla_rad
    nabla_rad = 3/(16*np.pi*a*c) * P_c * kappa * eps * m / (G * T_c**4 * m)


    #calculate luminosity
    l_new = eps * m
    #calculate radius
    r_new = (3 * m/(4 * np.pi * rho_c))**(1/3)
    #calculate pressure
    P_new = P_c - (3*G/(8*np.pi)) * (m**(2/3)) * ((4*np.pi/3 * rho_c)**(4/3)) 

    #calculate temperature
    #convective case
    T_new_conv_4 = T_c**4 - (1/(2*a*c) * (3/(4*np.pi))**(2/3) * kappa * eps * rho_c**(4/3) * m**(2/3))
    T_new_conv = T_new_conv_4**(1/4)
    #radiative case
    ln_T_new_rad = np.log(T_c) - ((np.pi/6)**(1/3) * G * nabla_ad * rho_c**(4/3) * m**(2/3) / P_c)
    T_new_rad = np.exp(ln_T_new_rad)

    #determine whether to use convective or radiative temperature
    if nabla_rad <= nabla_ad:
        #radiative case
        T_new = T_new_rad
    else:
        #convective case
        T_new = T_new_conv

    return l_new, P_new, r_new, T_new

def load2(m, L_star, R_star):
    #total luminosity, radius given as initial guess
    l_new = L_star
    r_new = R_star
    #calculate temperature
    T_new = (L_star/(4*np.pi*r_new**2 * sigma))**(1/4)
    #determine density
    def func(rho):
        #pressure through opacity function
        P1 = G*m/r_new**2 * 2/(3*opacity(rho, T_new))
        #pressure through EOS
        P2 = rho*N_A*k_B*T_new/mu + a*T_new**4/3
        return 1 - P2/P1
    #find density that satisfies both pressure equations
    rho_new = fsolve(func, 10**-7)[0]
    #print('rho',rho_new)

    #calculate pressure with obtained density
    P_new = rho_new * N_A * k_B * T_new / mu + a * T_new**4 / 3
    return l_new, P_new, r_new, T_new

def derivs(m, vars):
    l, P, r, T = vars
    rho = calculate_density(X,Y,Z,T,P)[0]
    beta = calculate_density(X,Y,Z,T,P)[1]
    eps = energy_generation(T, rho, X, Y, Z)
    #calculate dP/dM
    dP_dM = -G*m/(4*np.pi*r**4)
    #calculate dr/dM
    dr_dM = 1/(4*np.pi*r**2 * rho)
    #calculate dL/dM
    dl_dM = eps
    #calculate dT/dM
    #determine adiabatic gradients
    nabla_ad = 2*(4- 3*beta)/(32-24*beta-3*beta**2)
    #nabla_ad = 0.4
    nabla_rad = 3/(16*np.pi*a*c) * P * opacity(rho, T) * l / (G * T**4 * m)
    #determine whether to use convective or radiative temperature gradient
    if nabla_rad <= nabla_ad:
        #radiative gradient
        dT_dM = -G*m*T/(4*np.pi*r**4*P) * nabla_rad
    else:
        #convective gradient
        dT_dM = -G*m*T/(4*np.pi*r**4*P) * nabla_ad

    return dl_dM, dP_dM, dr_dM, dT_dM


