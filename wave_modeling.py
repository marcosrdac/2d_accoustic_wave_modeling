from time import time
import numpy as np
from numba import jit, cuda
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve as convolve
#from numpy import convolve
from scipy.signal import unit_impulse as delta
from scipy.ndimage import laplace as laplacian
#import matplotlib.animation as manimation
#import seaborn as sns
#from numba import jit, prange, typeof
#from  os import system
#from velocities import mean_V
#end = time()
#print('Bibs imported; time spent:', end-start)

LAPLACIAN_OPERATOR = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]]) / 6


import matplotlib.colors as colors
def plot(grid, fig, ax, contourf=True, four=None, phase=None, title=None,
        vmin=None, vmax=None,
        log=False):
    ax.set_title(title)
    if four:
        dfour  = np.fft.fft2(grid)
        dfour  = np.fft.fftshift(dfour)
        if phase:
            grid   = np.angle(dfour)
        else:
            grid   = np.abs(dfour)
        y = np.arange(-grid.shape[0]//2,
                      -grid.shape[0]//2 + grid.shape[0])
        x = np.arange(-grid.shape[1]//2,
                      -grid.shape[1]//2 + grid.shape[1])
    else:
        y = np.arange(grid.shape[0])
        x = np.arange(grid.shape[1])
    x,y = np.meshgrid(x,y)

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if contourf:
        ax.invert_yaxis()
        contourf = ax.contourf(x,y,grid, norm=norm)
        cbar = fig.colorbar(contourf, ax=ax)
    else:
        if log:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax, norm=norm,
                    cbar=False)
        else:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax)


def gen_3lay_v(nx, nz, h1, h2, V1, V2, V3,):
    '''
    h1 and h2 must each be lesser than (nz-amort_lay_len)/2
    '''
    h2 += h1
    v = np.empty((nz,nx), dtype=np.float64)
    v[  :h1, :] = V1
    v[h1:h2, :] = V2
    v[h2:  , :] = V3
    return(v)

#@cuda.jit("f8(f8[:,:],i8,i8)", nopython=True, parallel=True)
#@jit("f8[:,:](f8[:,:],f8,f8,i8,i8,i8)", nopython=True, parallel=True)
@jit("f8[:,:](f8[:,:],f8,f8,i8,i8,i8)", parallel=True)
def get_seis(v, dt=0.0001, h=1.,
        nz=300, nx=400, nt=1000):
    seis = np.empty((nt,nx))
    seis[0,:] = 0
    vdtoh2 = v*dt/h**2

    P0 = np.zeros_like(v)
    P = np.empty((nz, nx, 3))
    for i in range(3):
        P[:,:,i] = P0

    for i in range(1,nt):
        frame, oldframe, oldoldframe = i%3, (i-1)%3, (i-2)%3
        P[1,nx//2,oldframe] = np.exp(-i) * np.cos(30*i/nt)
        P[:,:,frame] = 2*P[:, :, oldframe] - P[:, :, oldoldframe] + \
                    vdtoh2 * convolve(
                            P[:, :, oldframe],
                            LAPLACIAN_OPERATOR,
                            'same')
        seis[i,:] = P[0,:,frame]
        #if i%5 == 0:
        #    plt.imshow(P[:,:,frame])
        #    plt.show()
    return(seis)



# v dt / h < 0.5
if __name__ == '__main__':
    NT = 1000
    NZ, NX = 300, 400
    h1, h2 = int(NZ//20), int(NZ//20)
    V1=3000
    V2=3000
    V3=3000
    v = gen_3lay_v(NX,NZ,h1,h2,V1,V2,V3)
    h = 1.
    dt = 0.0001
    START = time()
    seis = get_seis(v, dt, h, NZ, NX, NT)
    END = time()
    print(END-START)
    #plt.imshow(v[:,:])
    #plt.show()


    fig,ax = plt.subplots()
    plot(seis, fig, ax)
    plt.show()
    #plt.imshow(seismogram)
    #plt.show()
