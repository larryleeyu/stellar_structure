import numpy as np
#define constants in cgs units
N_A = 6.022e23 # Avogadro's number
k_B = 1.380649e-16 # Boltzmann constant in erg/K
a = 7.5657e-15 # radiation constant in erg/cm^3/K^4

def calculate_density(X, Y, Z, T, P):
    # calculate ionization mean molecular weight
    mu_ion = 1 / (X + Y/4 + Z/14)
    # calculate electron mean molecular weight
    mu_e = 1 / (X + Y/2 + Z/2)
    # calculate total mean molecular weight
    mu = 1 / (1/mu_ion + 1/mu_e)
    #mu = 4/(3 + 5*X)

    # calculate radiation pressure then subtract it from total pressure
    P_rad = a * T**4 / 3
    P_gas = P - P_rad

    # calculate density using ideal gas law
    rho = P_gas * mu / (N_A * k_B * T)
    #print ('For X =', X, ', Y =', Y, ', Z =', Z, ', log T =', np.log10(T), ', log P =', np.log10(P))
    #print('Density (g/cm^3):', rho)
    #print('Gas Pressure / Total Pressure:', P_gas/P)
    #print('-----------------------------------')
    return rho, P_gas/P

calculate_density(0, 0.98, 0.02, 10**7.55, 10**16.85)
calculate_density(0.7, 0.28, 0.02, 10**6.91, 10**16.87)

'''
For X = 0 , Y = 0.98 , Z = 0.02 , log T = 7.55 , log P = 16.85
Density (g/cm^3): 30.335335757424946
Gas Pressure / Total Pressure: 0.9435416916064454
-----------------------------------
For X = 0.7 , Y = 0.28 , Z = 0.02 , log T = 6.91 , log P = 16.87
Density (g/cm^3): 67.64156495872439
Gas Pressure / Total Pressure: 0.9998514995185559
-----------------------------------
'''