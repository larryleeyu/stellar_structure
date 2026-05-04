import pandas as pd
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np


#load in solar opacity data
df = pd.read_csv('opacity_0.02.txt',sep='\\s+',header=None)
#create opacity grid as 19x70 numpy array, replace 9.999 with np.nan
master_values = []
for i in range(19):
    arr= df[i].values
    for j in range(len(arr)):
        if arr[j] == 9.999:
            arr[j] = np.nan
            #arr[j] = 0
    master_values.append(arr)
master_values = np.array(master_values)

#create 19x70 temp grid, repeating the same 70 temps for each of the 19 rows
#given irregular temp values, load custom temp grid from file
log_temp = np.loadtxt('log_temps.txt')
temps = 10**log_temp

master_temps = []
for i in range(19):
    master_temps.append(temps)
master_temps = np.array(master_temps)

#create 19x70 density grid, using the formula rho = R*(T/10^6)^3, where R is the 19 values from log_R
log_R = np.linspace(-8,1,19)
R_values= 10**log_R 
master_rho = []
for i in range(len(R_values)):
    arr = []
    for j in range(len(temps)):
        rho = R_values[i]*(temps[j]/10**6)**3
        arr.append(rho)
    
    master_rho.append(arr)
master_rho = np.array(master_rho)


#flatten values for interpolation (work in log10 space to avoid scale issues)
points = np.column_stack((np.log10(master_rho.flatten()), np.log10(master_temps.flatten())))
values = master_values.ravel()

#filter nan entries
mask = ~np.isnan(values)
points = points[mask]
values = values[mask]

#evaluate opacity with interpolation in log-log space
q_rho_a, q_temp_a = 10**0.3, 10**6.3
q_rho_b, q_temp_b = 10**-4, 10**5.0
opacity_at_point_a = griddata(points, values, (np.log10(q_rho_a), np.log10(q_temp_a)), method='cubic')
opacity_at_point_b = griddata(points, values, (np.log10(q_rho_b), np.log10(q_temp_b)), method='cubic')
#print('Opacity for a:', opacity_at_point_a)
#print('Opacity for b:', opacity_at_point_b)

#determine opacity for given density and temperature by interpolating in log-log space
def opacity(rho, temp):
    log_rho = np.log10(rho)
    log_temp = np.log10(temp)
    return griddata(points, values, (log_rho, log_temp), method='cubic')
#print('Opacity for a:', opacity(q_rho_a, q_temp_a))
#print('Opacity for b:', opacity(q_rho_b, q_temp_b))
