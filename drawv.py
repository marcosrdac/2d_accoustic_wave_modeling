from wave_modeling import gen_3lay_v, draw_v
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


fig, ax = plt.subplots()

v = gen_3lay_v(300,200, 40,100,2700,3800,5200)

v = draw_v(ax, v, 25)
fig.savefig('example_model.png')
plt.show()
