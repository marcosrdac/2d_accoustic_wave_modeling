import numpy as np

a = np.load('/home/marcosrdac/projects/accoustic_wave_modelling/new_tests/h1_40 h2_100 V1_2700 V2_3000 V3_5700 nx_200 nz_200 amll_100 amort_0.023f.npy')[:1000]
#a = np.abs(a)
#a = np.exp(a)-1
b = np.load('/home/marcosrdac/projects/accoustic_wave_modelling/new_tests/h1_40 h2_100 V1_2700 V2_3000 V3_5700 nx_200 nz_200 amll_100 amort_0.023o.npy')[:1000]
#b = np.exp(a)-1
#b = np.abs(b)

print(np.linalg.norm(a-b))
