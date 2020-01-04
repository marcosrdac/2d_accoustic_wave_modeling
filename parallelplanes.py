import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_v(h1, h2, V1, V2, V3, nx, nz, amort_lay_len):
    '''
    h1 and h2 must each be lesser than (nz-amort_lay_len)/2
    '''
    H = nz - amort_lay_len
    v = np.empty((nz,nx), dtype=np.float64)
    v[  :h1, :] = V1
    v[h1:h2, :] = V2
    v[h2:  , :] = V3
    return(v)

nx, nz = 150, 120
amort_lay_len = 50
H = nz - amort_lay_len

hp1, hp2 = .3, .5

h1, h2 = int(hp1*H), int(hp2*H)
V1, V2, V3 = 1000,1500,3000

v = generate_v(h1, h2, V1, V2, V3, nx, nz)
#fig = plt.figure()
#ax  = fig.add_axes([0,0,1,1])
#sns.heatmap(v, ax=ax)
sns.heatmap(v,)
#fig.show()
plt.show()
