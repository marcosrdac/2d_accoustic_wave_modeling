from scipy.optimize import differential_evolution as diff_evo

def cost(x, sm1, sm2):
    u = x[0]
    v = x[1]
    return(u**2+v**2)

bounds = [(-1,1), (-1,1)]
args =   (1,2)

maxiter = 6
evo = diff_evo(cost, bounds, args, maxiter=maxiter)
found_params = evo.x

print(f'Found params: {found_params}')
