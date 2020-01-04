import numpy as np

velocities = {'dry sand':        [ .2, 1  ],
        'saturated sand':   [1.5, 2  ],
        'clay':             [1  , 2.5],
        'glacial till':     [1.5,2.5],
        'permafrost': [3.5,4.0],
        'sandstone': [2,6],
        'limestone': [2,6],
        'dolomites': [2.5,6.5],
        'salt': [4.5,5],
        'anhydrite': [4.5,6.5],
        'gypsum': [2,3.5],
        'granite': [5.5,6.0],
        'gabbro': [6.5,7],
        'ultramafic rocks': [7.5,8.5],
        'serpentinite': [5.5,6.5],
        'air': [.3,.3],
        'water': [1.4,1.5],
        'ice': [3.4,3.4],
        'petroleum': [1.3,1.4],
        'steel': [6.1,6.1],
        'iron': [5.8,5.8],
        'aluminium': [6.6,6.6],
        'concrete': [3.6],}

mean_V = {}
for i in velocities.keys():
    mean_V[i] = 1000*np.mean(np.array(velocities[i]))

if __name__ == '__main__':
    print(mean_V)
