import numpy as np

#determine energy generation rate from pp chain given temperature, density and composition
def pp_rate(T, rho, X, Y, Z):
    #temperatures
    T9 = T / 1e9
    T7 = T / 1e7
    #calculation for psi, estimation given 0.1 < Y < 0.5
    if T7 <= 1:
        psi = 1
    elif 1 < T7 < 2:
        psi = 0.5*T7 + 0.5
    else:
        psi = 1.5
    #calculations for f11 with weak screening
    zeta = (1*2/1) * X + (2*3/4) * Y + (7*8/14) * Z
    ED_kT = 5.92 * 10**(-3) * (zeta * rho/(T7**3))**(1/2)
    f11 = np.exp(ED_kT)
    #factor given in 18.63
    g11 = 1 + 3.82 * T9 + 1.51 * T9**2 + 0.144 * T9**3 - 0.0114 * T9**4
    #eq 18.63
    rate = 2.57 * 10**4 * psi * f11 * g11 * rho * X**2 * T9**(-2/3) * np.exp(-3.381 / T9**(1/3))
    return rate
#energy generation rate from CNO cycle given temperature, density and composition
def cno_rate(T, rho, X, Y, Z):
    #temperature
    T9 = T / 1e9
    #X_CNO set to Z/2 as recommended in stellar interiors
    X_CNO = Z / 2
    #factor given in 18.65
    g141 = 1 - 2.00 * T9 + 3.41 * T9**2 - 2.43 * T9**3
    #eq 18.65
    rate = 8.24 * 10**25 * g141 * X_CNO * X * rho * T9**(-2/3) * np.exp(-15.231 * T9**(-1/3) - (T9 / 0.8)**2)
    return rate
#total energy given as sum of pp and CNO contributions
def energy_generation(T, rho, X, Y, Z):
    pp_val = pp_rate(T, rho, X, Y, Z)
    cno_val = cno_rate(T, rho, X, Y, Z)
    #sum of the two rates
    return pp_val + cno_val

#print(pp_rate(10**(7.25), 10**(1.90), 0.70, 0.28, 0.02)/cno_rate(10**(7.25), 10**(1.90), 0.70, 0.28, 0.02))
#print(energy_generation(10**(7.25), 10**(1.90), 0.70, 0.28, 0.02))