# Module Created for Accoustic Seismic Modelling
# Author: Marcos Reinan de Assis Conceição
# e-mail: marcosrdac@gmail.com


from time import time
start = time()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import seaborn as sns
from numba import jit, prange, typeof
from  os import system
from velocities import mean_V
end = time()
print('Bibs imported; time spent:', end-start)


amst0 = 1e-14

@jit("f8(f8[:,:],i8,i8)", nopython=True, parallel=True)
def ortho_h2laplacian(A,x,z):
    '''
    Calculates the laplacian of a point in a 2D matrix.
    A: matrix
    x: axis 1 point position
    z: axis 0 point position
    '''
    return(
        # ortho = 1
            A[z-1, x ] + \
            A[z+1, x ] + \
            A[ z ,x-1] + \
            A[ z ,x+1] - \
        # point: -4
            4*A[z,x]
          )

@jit("f8(f8[:,:],i8,i8)", nopython=True, parallel=True)
def full_h2laplacian(A,x,z):
    '''
    Calculates the laplacian of a point in a 2D matrix.
    A: matrix
    x: axis 1 point position
    z: axis 0 point position
    '''
    return(
        # + ortho = 1/(1+1/sqrt2)
        0.58578644 * \
            (A[z-1, x ] + \
             A[z+1, x ] + \
             A[ z ,x-1] + \
             A[ z ,x+1]) + \
        # x diag = sqrt2/(1+1/sqrt2) / 2
        0.20710678 * \
            (A[z-1,x-1] + \
             A[z+1,x+1] + \
             A[z-1,x+1] + \
             A[z+1,x-1]) - \
        # point: 4*(ortho+diag)*point_value
        3.171572875*A[z,x]
          )

h2laplacian = ortho_h2laplacian

@jit("f8(f8[:,:,:],f8[:,:],f8,i8,i8,i8)", nopython=True, parallel=True)
def new_p(P, v, dtoh2, x, z, t):
    '''
    Calculates the next pressure value for a point in a pressure field matrix.
    P: pressure field matrix.
    v: velocity field matrix.
    dtoh2: time interval between frames times distance between points in
        velocity and pressure matrix.
    '''
    return(2*P[t-1,z,x] - P[t-2,z,x] + \
             dtoh2*v[z,x]**2 * h2laplacian(P[t-1],x,z))

@jit("f8(f8[:,:,:],f8[:,:],f8,f8,i8,i8,i8)", nopython=True, parallel=True)
def new_amort_p(P, v, dtoh2, w, x, z, t):
    '''
    Calculates the next pressure value for a point in the damped zone of a
    pressure field matrix.
    P: pressure field matrix.
    v: velocity field matrix.
    dtoh2: time interval between frames times distance between points in
        velocity and pressure matrix.
    w: damping coeficient.
    '''
    return(2*P[t-1,z,x] - P[t-2,z,x] + \
                w*dtoh2*v[z,x]**2 * h2laplacian(P[t-1],x,z))

@jit("i8(i8,i8,i8,i8,i8)", nopython=True, parallel=True)
def nearest_border_distance(nx, nz, border, x, z):
    '''
    Calculates the distance between a point and a border of a matrix.
    nz: size of matrix along z axis.
    nx: size of matrix along x axis.
    border: number that marks the position of the border considering the numpad
    arrow keys.
    x, z: point position.
    '''
    # orthogonal
    if border==4: return(x)
    if border==8: return(z)
    if border==2: return(nz-z-1)
    if border==6: return(nx-x-1)
    #diagonal
    if border==7: return(min(x   , z   ))
        # bellow is 6ms slower if nx,nz=300,250, idkw
        #if border==7: return(np.amin(np.array([x   , z   ], np.int64)))
        # so I'm using the built-in function min
    if border==9: return(min(nx-x-1, z   ))
    if border==1: return(min(x   , nz-z-1))
    if border==3: return(min(nx-x-1, nz-z-1))
    else: return(0)

@jit("f8(f8,i8,i8)", parallel=True)
def calc_amort_factor(amort, amort_lay_len, border_distance):
    '''
    amort: measure of damping applied.
    amort_lay_len: size of damping layer.
    border_distance: distance to nearest border.
    '''
    return(np.exp(-(amort * np.float64(amort_lay_len-border_distance))**2))

@jit(nopython=True, parallel=True)
def calculate_P(P0,
                v, dt=.0001, nt=600,
                h=1., sx=0, sz=0,
                signal=np.array([]),
                amort_lay_len=40,
                amort=1.,
                amort_top = False,
                amort_bottom = False,
                amort_left = False,
                amort_right = False,):
    '''
    Calculate pressure field over time.

    Please let v*dt/h <= .4

    P0: initial pressure conditions.
    v: velocity field matrix.
    sx, sz: source position.
    nt: number of frames calculated.
    dt: time interval between frames.
    h: distance of points in matrix.
        I must say results are better when V/h ~= 400 (+-200), so h ~= V/400
    amort_lay_len: size of damping layer.
    amort: measure of damping applied.
    amort_top, amort_bottom, amort_left, amort_right: whether or not to put a
        damping layer at the top, bottom, left or right border of the matrix.
    '''

    nz, nx = P0.shape
    signal_len = len(signal)
    dtoh2 = (dt/h)**2

    # defining loop intervals
    # non-damped intervals
    nT_x = np.empty(2, dtype=np.int64)
    nT_z = np.empty(2, dtype=np.int64)
    if amort_left:   nT_x[0] = amort_lay_len
    else:            nT_x[0] = 1
    if amort_right:  nT_x[1] = nx-amort_lay_len
    else:            nT_x[1] = nx-1
    if amort_top:    nT_z[0] = amort_lay_len
    else:            nT_z[0] = 1
    if amort_bottom: nT_z[1] = nz-amort_lay_len
    else:            nT_z[1] = nz-1
    # orthogonal damped intervals
    a4T_x = np.array([1,    nT_x[0]], dtype=np.int64)
    a4T_z = nT_z
    a8T_x = nT_x
    a8T_z = np.array([1,    nT_z[0]], dtype=np.int64)
    a6T_x = np.array([nT_x[1], nx-1], dtype=np.int64)
    a6T_z = nT_z
    a2T_x = nT_x
    a2T_z = np.array([nT_z[1], nz-1], dtype=np.int64)
    #  diagonal damped intervals
    a1T_x = np.array([1,    nT_x[0]], dtype=np.int64)
    a1T_z = np.array([nT_z[1], nz-1], dtype=np.int64)
    a7T_x = np.array([1,    nT_x[0]], dtype=np.int64)
    a7T_z = np.array([1,    nT_z[0]], dtype=np.int64)
    a3T_x = np.array([nT_x[1], nx-1], dtype=np.int64)
    a3T_z = np.array([nT_z[1], nz-1], dtype=np.int64)
    a9T_x = np.array([nT_x[1], nx-1], dtype=np.int64)
    a9T_z = np.array([1,    nT_z[0]], dtype=np.int64)


    # creating pressure field
    P = np.empty((nt,nz,nx), dtype=np.float64)
    P[ 0 ] = P0
    P[-1 ] = P0

    def calculate_nPn(nT_x, nT_z, t):
        '''
        Calculates the non-damped part of frame t of the pressure matrix.
        '''
        for x in prange(nT_x[0], nT_x[1]):
            for z in prange(nT_z[0], nT_z[1]):
                P[t,z,x] = new_p(P, v, dtoh2, x, z, t)

    def calculate_aPn(aT_x, aT_z, border, t):
        '''
        Calculates the damped part of frame t of the pressure matrix.
        '''
        for x in prange(aT_x[0],aT_x[1]):
            for z in prange(aT_z[0],aT_z[1]):
                w = calc_amort_factor(amort, amort_lay_len,
                                      nearest_border_distance(nx, nz,
                                                              border,
                                                              x, z))
                P[t,z,x] = new_amort_p(P, v, dtoh2, w, x, z, t)

    def calculate_full_frame(t):
        '''
        Calculates the entire frame t of the pressure field.
        '''
        # no amort zone
        calculate_nPn(nT_x,nT_z,t)
        # amort zone
        #  orthogonal
        calculate_aPn(a4T_x, a4T_z, 4, t)
        calculate_aPn(a6T_x, a6T_z, 6, t)
        calculate_aPn(a2T_x, a2T_z, 2, t)
        calculate_aPn(a8T_x, a8T_z, 8, t)
        #  diagonal
        calculate_aPn(a1T_x, a1T_z, 1, t)
        calculate_aPn(a3T_x, a3T_z, 3, t)
        calculate_aPn(a7T_x, a7T_z, 7, t)
        calculate_aPn(a9T_x, a9T_z, 9, t)
        # non-reflexive borders (not anymore)
        #for z in range(nT_z[0], nT_z[1]):
        #    P[t,z,nx-1] = (P[t-1,z,nx-3]-P[t-1,z,nx-2])*v[z,nx-1]*dt/h + P[t-1,z,nx-1]

    for t in range(1, nt):
        if t <= signal_len:
            P[t-1,sz,sx] = signal[t-1]  # correct
        calculate_full_frame(t)

    # bellow is slow for some reason :o
    # test for various nt values
    #for t in range(1, signal_len+1):
    #    P[t,sz,sx] = signal[t-1]  # correto
    #    calculate_nPn(nT_x,nT_z)
    #for t in range(signal_len+1, nt):
    #    calculate_nPn(nT_x,nT_z)
    return(P)

def save_gif(P, name='animation',
             v=None, V=300,
             T=None,
             fps=16,
             vmin=None, vmax=None,
             vwin=None,
             cmap='rainbow', dpi=150):
    '''
    Saves a [t,z,x] array as a gif.
    '''
    filename = name + '.gif'

    frames_div = 5

    nt, nz, nx = P.shape
    fig = plt.figure(dpi=dpi)
    fig.set_size_inches(2*nx/np.linalg.norm([nx,nz]),
                        2*nz/np.linalg.norm([nx,nz]))

    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    ax.set_xticks([])
    ax.set_yticks([])

    if vwin:
        for coord in ['xi', 'zi', 'xf', 'zf']:
            if not coord in vwin.keys(): vwin[coord] = None
        vview = P[:,vwin['zi']:vwin['zf'],vwin['xi']:vwin['xf']]
        vmin = vview.min()
        vmax = vview.max()
    else:
        if not vmin: vmin = P.min()
        if not vmax: vmax = P.max()

    frames = []
    for frame in P[1::frames_div]:
         frames.append(
             [ax.imshow(frame, cmap, animated=True, vmin=vmin, vmax=vmax,
                 interpolation='nearest')])

    start = time()
    gif = manimation.ArtistAnimation(fig, frames, blit=True)
    gif.save(filename, writer='imagemagick', fps=fps)
    end = time()
    print('Gif saved; time spent:', end-start, end='.\n\n')
    plt.close(fig)

    return(P)

def where_to_amort(where):
    '''
    This function was made to ease the writting of damping positions to give as
    argument to calculate P.
    '''
    where_to_amort = {'amort_top':    False,
                      'amort_bottom': False,
                      'amort_left':   False,
                      'amort_right':  False}
    if 'top' in where_to_amort:
        where_to_amort['amort_top'] = True
    if 'bottom' in where:
        where_to_amort['amort_bottom'] = True
    if 'left' in where:
        where_to_amort['amort_left'] = True
    if 'right' in where:
        where_to_amort['amort_right'] = True
    if 'vertical' in where:
        where_to_amort['amort_top'] = True
        where_to_amort['amort_bottom'] = True
    if 'horizontal' in where:
        where_to_amort['amort_left'] = True
        where_to_amort['amort_right'] = True
    if 'all' in where:
        where_to_amort['amort_top'] = True
        where_to_amort['amort_bottom'] = True
        where_to_amort['amort_left'] = True
        where_to_amort['amort_right'] = True
    return(where_to_amort)

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

def draw_v(ax, v, xticks=50, Vmin=200, Vmax=8500):
    ax.invert_yaxis
    ax.xaxis.tick_top()
    sns.heatmap(v, ax=ax, vmin=Vmin, vmax=Vmax, xticklabels=xticks)


def propagate_signal(v, dt=.001, nt=600,
                     h=10., sx=0, sz=0,
                     signal=np.array([]),
                     amort_lay_len=50,
                     amort=0.045,
                     where='',):
    P0 =            np.zeros(v.shape)
    start = time()  # recording time
    P = calculate_P(P0=P0,
                    v=v, dt=dt, nt=nt,
                    h=h, sx=sx, sz=sz,
                    signal=signal,
                    amort_lay_len=amort_lay_len,
                    amort=amort,
                    **where_to_amort(where),)
    end = time()    # recording time
    #print('Frames calculated; time spent:', end-start)
    return(P)

def extract_seis(P,spacing=1, crop_left=None, crop_right=None):
    nx = P.shape[2]
    if not crop_left: crop_left = 0
    if not crop_right: crop_right = nx
    seis = P[:,1,crop_left:-crop_right:spacing]
    return(seis)

def draw_seis(ax, seis, spacing=1, amp=1000, max_value=None,
              xi=None, xf=None):
    if not max_value: max_value=seis.max()
    nt, nx = seis.shape
    if not xi: xi = 0
    if not xf: xf = nx
    if xf < 0: xf = nx-abs(xf)

    seis *= amp/max_value
    seisp = seis.clip(min=0, max=spacing)
    seisn = seis.clip(max=0, min=-spacing)
    y = np.arange(nt)
    for x in range(0, nx, spacing):
        ax.fill(x+spacing*seisp[:,x], y, 'red')
        ax.fill(x+spacing*seisn[:,x], y, 'blue')
    ax.invert_yaxis()
    ax.set_xlim(xi,xf)

def central_shot():
    name =          'central'
    h =             10
    nx =            200
    nz =            150
    amort_lay_len = 50
    nx += 2*amort_lay_len
    nz +=   amort_lay_len

    sx =            nx//2
    sz =            1
    amort =         0.045
    where =         'bottom horizontal'

    H =             nz-amort_lay_len
    h1 =            int(.1*H)
    h2 =            int(.4*H)
    V1 =            mean_V['dry sand']
    V2 =            mean_V['clay']
    #V3 =            mean_V['salt']
    V3 =            mean_V['clay']
    dt = .001
    nt = 2500
    v =  gen_3lay_v(nx, nz,
                    h1, h2,
                    V1, V2, V3,)

    signal_len = int(0.56/dt)  # .56 ms was choosen arbitrarily
    start = time()
    with np.errstate(divide='ignore'):
        x = np.linspace(0,3,signal_len)
        signal = x**(1/x)*np.exp(-x**2)
    end = time()
    print('Signal calculated; time spent:', end-start)

    start = time()  # recording time
    P = propagate_signal(v, dt=dt, nt=nt,
                         h=h, sx=sx, sz=sz,
                         signal=signal,
                         amort_lay_len=amort_lay_len,
                         amort=amort,
                         where=where,)
    end = time()    # recording time
    print('Frames calculated; time spent:', end-start)

    #save_gif(P[::5,:,:], name)

    spacing = 1
    seis = extract_seis(P, spacing=spacing, crop_left=amort_lay_len,
            crop_right=amort_lay_len)
    np.save(name, seis)
    del(P)

    #seis = np.load(name+'.npy')

    fig, ax = plt.subplots(1,1)
    draw_seis(ax, seis, spacing=spacing, amp=40,)
    fig.savefig(name+'.png')
    #fig.show()

    #sns.heatmap(seis)

def get_signal(dt):
    signal_len = int(0.3/dt)  # .56 ms was choosen arbitrarily
    start = time()
    with np.errstate(divide='ignore'):
        x = np.linspace(0,3,signal_len)
        signal = x**(1/x)*np.exp(-x**2)
    end = time()
    #print('Signal calculated; time spent:', end-start)
    return(signal)

default_signal = get_signal(.001)

def get_central_seis(h1, h2, V1, V2, V3, signal=None,
                     dt=.001, h=10, nx=300, nz=300, nt=2000,
                     amort_lay_len=50, amort=.045,
                     spacing=1):
    name = f'h1_{h1} h2_{h2} V1_{V1} V2_{V2} V3_{V3} nx_{nx} nz_{nz}' + \
           f' amll_{amort_lay_len} amort_{amort}'
    if not signal: signal = default_signal
    nx +=         2*amort_lay_len
    nz +=           amort_lay_len
    sx =            nx//2
    sz =            1
    where =         'bottom horizontal'

    v =  gen_3lay_v(nx, nz,
                    h1, h2,
                    V1, V2, V3,)

    P = propagate_signal(v, dt=dt, nt=nt,
                         h=h, sx=sx, sz=sz,
                         signal=signal,
                         amort_lay_len=amort_lay_len,
                         amort=amort,
                         where=where,)

    #save_gif(P[::5,:,:], name)

    seis = extract_seis(P, spacing=spacing, crop_left=amort_lay_len,
            crop_right=amort_lay_len)
    np.save(name, seis)

    fig, ax = plt.subplots(dpi=300)
    draw_seis(ax, seis)
    fig.savefig(name+'.png')

    return(seis)

def get_seis(h1, h2, V1, V2, V3, signal=None,
                     dt=.001, h=10, nx=200, nz=200, nt=2000,
                     amort_lay_len=100, amort=.023,
                     spacing=5):
    if not signal: signal = get_signal(dt)
    nx +=         2*amort_lay_len
    nz +=           amort_lay_len
    sx =            nx//2
    sz =            1
    where =         'bottom horizontal'

    print(f'\th1={h1}, h2={h2}, V1={V1}, V2={V2}, V3={V3}')

    v =  gen_3lay_v(nx, nz,
                    h1, h2,
                    V1, V2, V3,)

    P = propagate_signal(v, dt=dt, nt=nt,
                         h=h, sx=sx, sz=sz,
                         signal=signal,
                         amort_lay_len=amort_lay_len,
                         amort=amort,
                         where=where,)

    seis = extract_seis(P, spacing=spacing, crop_left=amort_lay_len,
            crop_right=amort_lay_len)
    del(P)
    return(seis)


# v dt / h < 0.5
if __name__ == '__main__':
    dt = 0.001
    nt = 2500
    #nt = 1700
    #nt = 1600
    signal = get_signal(dt)

    nz = 200
    nx = 200
    amort_lay_len=100
    amort = .023
    #h1 = np.random.randint(0,nz//2)
    #h2 = np.random.randint(0,nz//2)
    #V1 = np.random.randint(200,8000)
    #V2 = np.random.randint(200,8000)
    #V3 = np.random.randint(200,8000)
    h1=45
    h2=65
    V1=3000
    V2=2500
    V3=5700

    seis = get_central_seis(h1, h2, V1, V2, V3,
            nz=nz, nx=nx, nt=nt,
            amort_lay_len=amort_lay_len, amort=amort)
