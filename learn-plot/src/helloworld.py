import numpy as np
import matplotlib.pyplot as plt

def h(p):
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

p = np.arange(0.01, 1, 0.01)

plt.plot(p, h(p))
plt.show()
