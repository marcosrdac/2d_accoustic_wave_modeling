import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#sns.set_context('paper')
x = np.linspace(0,3,1000)
x = x/10
with np.errstate(divide='ignore'):
    #signal = x**(1/x)*np.exp(-x**2)
    signal = (10*x)**(1/x/10)*np.exp(-x**2*100)

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(x,signal)
ax.set_xlabel('t (s)')
ax.set_ylabel('P (p.u.)')

fig.tight_layout()

fig.savefig('c.png')
plt.show()
