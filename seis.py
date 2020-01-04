import numpy as np
import scipy as scp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from time import time
from numba import jit
from wave_modeling7 import save_gif


@jit(nopython=True)
def h2laplacian(A,lx,lz):
    return(
        # ortho = 1/(1+1/sqrt2)
        0.58578644 * \
            (A[ lz ,lx+1] + \
             A[lz+2,lx+1] + \
             A[lz+1, lx ] + \
             A[lz+1,lx+2]) + \
        # diag = sqrt2/(1+1/sqrt2) / 2
        0.20710678 * \
            (A[ lz , lx ] + \
             A[lz+2,lx+2] + \
             A[ lz ,lx+2] + \
             A[lz+2, lx ]) - \
        # point: 4*(ortho+diag)*point_value
        3.171572875*A[lz+1,lx+1]
          )


@jit(nopython=True)
def get_P(P0, v, signal=np.array([]), sx=0, sz=0, nt=600, dt=.001, amort_lay_len=40, amort_top=True, h=1):
    amort = .098

    nz, nx = P0.shape
    P = np.zeros((nt,nz,nx))
    P[0] = P0
    P[-1] = P0
    signal_len = len(signal)
    dtoh2 = (dt/h)**2
    for (t,lz,lx), p in np.ndenumerate(P[1:,1:-1, 1:-1]):
        if t-1 < signal_len:
            P[t,sz,sx] = signal[t-1]
        if amort_top:
            in_border_value = min(lx+1, lz+1, nz-lz-1, nx-lx-1)
        else:
            in_border_value = min(lx+1, nz-lz-1, nx-lx-1)
        if in_border_value < amort_lay_len:
            w = np.exp(-(amort*(amort_lay_len-in_border_value))**2)
            #w = 1
        else:
            w = 1
        P[t,lz+1,lx+1] = 2*P[t-1,lz+1,lx+1] - \
                           P[t-2,lz+1,lx+1] + \
                               w * v[lz+1,lx+1]**2*dtoh2 * h2laplacian(P[t-1],lx,lz)
#             w = np.exp(-(amort*(5-lx))**2)
    return(P)


def get_pf(filename='gif.gif',
             P0=None, v=None,
             signal=np.zeros(1),
             sx=None, sz=None,
             nx=50, nz=50, nt=600,
             h=1, dt=0.001,
             T=None, amort_top=True,
             amort_lay_len=40,
             fps=25,
             V=300,
             vmin=None, vmax=None,
             vwin=None,
             cmap='rainbow'):
    '''
    This function calculates and returns the solution to bidimensional
    pressure field as an animated gif image based on the wave equation, which is
    approximated with the finite differences method.

    filename: file name to save gif.
    P0: initial pressure field;
    v: propagation velocity field;
    signal: wave source signal to be propagated;
    sx, sz: position to put source in;
    nx, nz: square mesh size;
    nt: number of frames to be calculated;
    h: distance between points in square mesh  # I MUST TAKE A LOOK AT IT SOMEDAY.
    dt: time interval between frames calculated;
    T: number of secconds the gif will have (it overwrites nt);
    fps: gif ammount of frames per seccond;
    V: if no velocity field is given, use a homogenous medium with wave propagation
        velocity given by V;
    vmin, vmax: min and max values in the scale of pressures to be plotted in the
        gif. If they are not given, consider the min and max of the pressure field
        calculated (for all frames).
    vwin: {'xi': xi, 'xf': xf, 'zi': zi, 'zf': zf} window of values to look for vmin
        and vmax. 'i' stands for initial, 'f' stands for final. setting vmin overwrites
        vmin and vmax previously set.
    cmap: cmap to be used in matplotlib's plots.
    '''

    frames_div = 5
    if T: nt = T*frames_div*fps


    there_is_P0 = type(P0)==np.ndarray
    there_is_v = type(v)==np.ndarray

    if there_is_P0:
        nz, nx = P0.shape
        if not there_is_v:
            v = V * np.ones((nz,nx))
    elif there_is_v:
        nz, nx = v.shape
        P0 = np.zeros((nz, nx))
    else:
        P0 = np.zeros((nz, nx))
        v = V * np.ones((nz,nx))
        signal=[0,1,0,-1,0]

    if not sx: sx = int(round(nx/2))
    if not sz: sz = int(round(nz/2))

    start = time()
    P = get_P(P0, v, signal, sx=sx, sz=sz, nt=nt, dt=dt, h=h, amort_lay_len=amort_lay_len,
             amort_top=amort_top)
    end = time()
    print('\tframes calculated; spent time:', end-start, end='.\n')

    return(P)


def get_seis(h1=20, h2=50, v1=180, v2=300, v3=700, nx=200, nz=150):
    h1_, h2_ = (nz/100*np.array([h1, h2])).astype(int)
    h3_ = nz-h1_-h2_
    dt=.001
    signal=np.exp(-1/(np.sin(np.linspace(0.002,np.pi-.002,100))))

    v = np.concatenate([np.tile(v1,h1_*nx).reshape(h1_,nx),
                        np.tile(v2,h2_*nx).reshape(h2_,nx),
                        np.tile(v3,h3_*nx).reshape(h3_,nx)],
                    axis=0)

    P = get_pf('3lay.gif',v=v, sx=None, sz=1, T=10,signal=signal, dt=dt, amort_top=False)
    return(P[:,1,:])


def custo(x, nx, nz, seis):
    h1 = x[0]
    h2 = x[1]
    v1 = x[2]
    v2 = x[3]
    v3 = x[4]
    seis_n = get_seis(h1, h2, v1, v2, v3, nx, nz)
    norm = np.linalg.norm(seis - seis_n)
    return(norm)


nx = 200
nz = 150
#
#
#seis = get_seis(h1=20, h2=50, v1=180, v2=300, v3=700, nx=nx, nz=nz)
#
#bounds = [[5,45], [5,45], [150, 800], [150, 800], [150, 800]]
#args = nx, nz, seis
#
#print(scp.optimize.differential_evolution(custo, bounds, args, maxiter=6).x)
#
##x = [22, 52, 200, 300, 700]
##c = custo(x,  nx, nz, seis)
##print(c)


#h1=20
#h2=50
#v1=180
#v2=300
#v3=700
#
#h1_, h2_ = (nz/100*np.array([h1, h2])).astype(int)
#h3_ = nz-h1_-h2_
#dt=.001
#signal=np.exp(-1/(np.sin(np.linspace(0.002,np.pi-.002,100))))
#
#v = np.concatenate([np.tile(v1,h1_*nx).reshape(h1_,nx),
#                    np.tile(v2,h2_*nx).reshape(h2_,nx),
#                    np.tile(v3,h3_*nx).reshape(h3_,nx)],
#                axis=0)
#
#P = get_pf('3lay.gif',v=v, sx=None, sz=1, T=10,signal=signal, dt=dt, amort_top=False)
#P = get_pf('3lay.gif',v=v, sx=None, sz=1, T=10,signal=signal, dt=dt, amort_top=False)
#save_gif(P)


amst0 =         1e-14
nx, nz, nt =    40,40,5000
sx, sz =        nx//2, nz//2
signal_len =    300 #100 #200 #nt//3 # if nt < 300 else 100
v =             signal_len//2 * np.ones((nz,nx))
P0 =            np.zeros(v.shape)
signal =        100*np.exp(-1/(np.sin(np.linspace(amst0, np.pi-amst0, signal_len))))
#signal =        np.ones(nt)
#P0[sz,sx] =    1
amort_lay_len = nz//2 #(nz-1)//2  # 2*nz//8
P = get_pf('3lay.gif', v=v, sx=sx, sz=sz, nt=nt,signal=signal, amort_top=False)
save_gif(P[::5])
