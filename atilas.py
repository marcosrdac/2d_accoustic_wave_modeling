#------------------------------------------------------------------------------+
#
#   Nathan A. Rooy, modified by Átila S. Quintela
#   A simple, bare bones, implementation of differential evolution with Python
#   August, 2017; modified March, 2018
#
#------------------------------------------------------------------------------+

#--- IMPORT DEPENDENCIES ------------------------------------------------------+

import random
from os.path import exists
from sys import argv
from numba import jit
import numpy as np
import pandas as pd
import scipy.optimize as sp

script, input_file, output_file = argv

# --- Example Cost functions -----------------------------------------------___+

def func1(x):
    # Sphere function, use any bounds, f(0,...,0)=0
    return sum([x[i]**2 for i in range(len(x))])

def func2(x):
    # Beale's function, use bounds= [(-4.5, 4.5),(-4.5, 4.5)], f(3,0.5)=0.
    term1 = (1.500 - x[0] + x[0]*x[1])**2
    term2 = (2.250 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3

def checktype(lista):

    chave = True
    tipo = type(lista[0])
    i = 1
    while (i < len(lista) and chave == True):
        aux = tipo
        tipo = type(lista[i])
        if tipo != aux:
            chave = False
        i = i + 1

    if chave == True:
        return "Homogênea"
    else:
        print(tipo)
        return "Heterogênea"

@jit(parallel=True)
def calc_f(param, perm_exp, por):
    zeta = param[0]
    eta = param[1]
    xi = param[2]

    #perm_calc = [por_i *  ( xi * ( ( (por_i)**((zeta + 2)/2) ) / ( (1 - por_i)**eta ) ) )**(2) for por_i in por]
    perm_calc = []
    for por_i in por:
        perm_calc.append((1 - por_i)**(-2 * eta) * xi*xi * por_i**(zeta + 3))
#    perm_calc = [(1 - por_i)**(-2 * eta) * xi*xi * por_i**(zeta + 3) for por_i in por]

    soma = 0.0
    for perm_calc_i, perm_exp_i in zip(perm_calc, perm_exp):
        soma = soma + (perm_calc_i - perm_exp_i)**2
    #soma = sum([(perm_calc_i - perm_exp_i)**2 for perm_calc_i, perm_exp_i in zip(perm_calc, perm_exp)])

    return soma


def calc_k(param, perm_exp, por):
#    print(param)
    zeta = param[0]
    eta = param[1]
    xi = param[2]


    perm_calc = [por_i *  ( xi * ( ( (por_i)**((zeta + 2)/2) ) / ( (1 - por_i)**eta ) ) )**(2) for por_i in por]
    return perm_calc
# ---- Functions ----------------------------------------------------------+

def ensure_bounds(vec, bounds):
    vec_new = []
    # cycle through each variable in vector
    for i in range(len(vec)):

        # variable exceeds the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceeds the maximum boundary
        elif vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # variable is fine
        else:
            vec_new.append(vec[i])

    return vec_new

def initialize(bounds,popsize):
    population = []
    for i in range(0,popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0],bounds[j][1]))
        population.append(indv)

    return population

def mutation(population, bounds, j):
    # select three random vector index positions [0, popsize), not including current vector (j)
    canidates = list(range(0,popsize))
    canidates.remove(j)
    random_index = random.sample(canidates, 3)

    x_1 = population[random_index[0]]
    x_2 = population[random_index[1]]
    x_3 = population[random_index[2]]
    x_t = population[j]     # target individual

    # subtract x3 from x2, and create a new vector (x_diff)
    x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

    # multiply x_diff by the mutation factor (F) and add to x_1
    v_donor = [x_1_i + (mutate * x_diff_i) for x_1_i, x_diff_i in zip(x_1, x_diff)]
    v_donor = ensure_bounds(v_donor, bounds)

    return v_donor, x_t

def recombinando(v_donor,x_t,recombination):
    v_trial = []
    # cycle through each variable in our target vector
    for k in range(len(x_t)):
        crossover = random.random()

        # recombination occurs when crossover <= recombination rate
        if crossover <= recombination:
            v_trial.append(v_donor[k])

        # recombination did not occur
        else:
            v_trial.append(x_t[k])

    return v_trial

def selection(gen_scores,population,v_trial,x_t,perm,por,j):
    score_trial = cost_func(v_trial, perm, por)
    score_target = cost_func(x_t, perm, por)


    if score_trial < score_target:
        population[j] = v_trial
        gen_scores.append(score_trial)
        print('  >',score_trial, v_trial)

    else:
        print('  >',score_target, x_t)
        gen_scores.append(score_target)


    return gen_scores,population

def score_keeping(population,gen_scores,popsize):
    gen_avg = sum(gen_scores) / popsize
    gen_best = min(gen_scores)
   # print("length of population",len(population))
    gen_sol = population[gen_scores.index(min(gen_scores))]

    print('     > GENERATION AVERAGE:', gen_avg)
    print('     > GENERATION BEST:', gen_best)
    print('     > BEST SOLUTION:', gen_sol, '\n')

    print(" -------------------------- ")

    return gen_sol, gen_best

def main(cost_func, bounds, popsize, mutate, recombination, maxiter, perm_exp, por, erro):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    population = initialize(bounds,popsize)

    #--- SOLVE --------------------------------------------+

    print('GENERATION:',1)

    # cycle through each generation (step #2)
    gen_scores = []

    # cycle through each individual in the population (step #3)
    for j in range(0, popsize):

        #--- MUTATION (step #3.A) ---------------------+

        v_donor, x_t = mutation(population, bounds, j)

        # ---- Recombination (step #3.B) ---------------#

        v_trial = recombinando(v_donor, x_t,recombination)

        # ---- Greedy selection (step #3.C) ------------------ +

        gen_scores, population = selection(gen_scores,population,v_trial,x_t,perm,por,j)


    # --- Score keeping
    gen_sol, gen_best = score_keeping(population,gen_scores,popsize)

    i = 2
    while (gen_best > erro and i <= maxiter):

        #--- SOLVE --------------------------------------------+

        print('GENERATION:',i)

        # cycle through each generation (step #2)
        gen_scores = []

        # cycle through each individual in the population (step #3)
        for j in range(0, popsize):

            #--- MUTATION (step #3.A) ---------------------+

            v_donor, x_t = mutation(population, bounds, j)

            # ---- Recombination (step #3.B) ---------------#

            v_trial = recombinando(v_donor, x_t,recombination)

            # ---- Greedy selection (step #3.C) ------------------ +
            gen_scores, population = selection(gen_scores,population,v_trial,x_t,perm,por,j)


        # --- Score keeping
        gen_sol, gen_best = score_keeping(population,gen_scores,popsize)

        i = i + 1
    return gen_sol,gen_best

# ---- constants ---------------- +

cost_func = calc_f              # Cost function
bounds = [(0.0,10.0),(0.0,10.0),(0.0,10.0)] # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
popsize = 100               # Population size, must be >= 4
mutate = 0.6                    # Mutation factor [0,2]
recombination = 0.6             # Recombination rate [0,1]
maxiter = 300                   # Max number of generations (maxiter)
erro = 10**(-3)

# ---- Run ------------------------ +

leitura_decodificada = pd.read_csv(input_file)

#por_total = list(leitura_decodificada['por_total'])
#por_efetiva = leitura_decodificada['por_e'].tolist()
#perm = leitura_decodificada['k(md)'].tolist()

por_total = list(leitura_decodificada['por'])
perm = leitura_decodificada['permd'].tolist()


# ---------------------  Utilizando meu próprio código --------------------------------- #
param, f = main(cost_func, bounds, popsize, mutate, recombination, maxiter, perm, por_total, erro)

print('Os parâmetros calculados são: ', param)
print('O erro dos parâmetros calculados é', f)

print("Vamos plotar os valores da permeabilidade resultantes do algorítimo do desse script")
k = calc_k(param,perm, por_total)
for k_i in k:
    print(k_i)

try:
    saida = open(output_file,'w')
    for por_i,k_i in zip(por_total,k):
        saida.write(f"{por_i},{k_i} \n")
    saida.write("\n\n")
    saida.write(f"""
zeta,{param[0]}
eta,{param[1]}
xi,{param[2]}""")

finally:
    saida.close()




# ----------------------  Utilizando Algoritimo do SciPy--------------------------------- #
arg = perm, por_total
result = sp.differential_evolution(calc_f,bounds,arg,'best1bin',maxiter,popsize,0.01,mutate,recombination)

print("Vamos plotar os valores da permeabilidade resultantes do algorítimo do scipy")
k = calc_k(result.x,perm, por_total)
for k_i in k:
    print(k_i)

print('Os parâmetros calculados são: ',result.x)
print('O erro dos parâmetros calculados é',calc_f(result.x,perm, por_total))
# ---- End ------------------------- +
