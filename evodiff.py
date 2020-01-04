import numpy as np
import scipy as scp
from wave_modeling import get_seis
from time import time


        #(h1,h2,V1,V2,V3)
params = (45,65,3000,2500,5700)
nt = 1650

print('Starting Genetic Algorithm')
ref_seis = get_seis(*params, nt=nt)
print()


def cost(x, ref_seis, nt):
    h1 = int(x[0])
    h2 = int(x[1])
    V1 = x[2]
    V2 = x[3]
    V3 = x[4]

    seis = get_seis(h1,h2,V1,V2,V3,nt=nt)

    dif_norm = np.linalg.norm(seis - ref_seis)
    print(f'\t\tDifference norm: {dif_norm}')
    return(dif_norm)


args = (ref_seis, nt)
bounds = [[  0, 100 ],
          [  0, 100 ],
          [200, 8000],
          [200, 8000],
          [200, 8000]]

maxiter = 50


print('Starting evolution')
start = time()
evo_diff = scp.optimize.differential_evolution(cost, bounds, args,  maxiter=maxiter, disp=True)
end = time()
print('Evolution cessed; time spent: {end-start}')

found_params = evo_diff.x
param_errors = np.array(found_params) - np.array(params)

print()
print(f'Found params:\n{found_params}\n')
print(f'Param errors: {param_errors}')
