import numpy as np

from cec2017.functions import f5

samples = 3
dimension = 50
x = np.random.uniform(-100, 100, size=(samples, dimension))
val = f5(x)
for i in range(samples):
    print(f"f5(x_{i}) = {val[i]:.6f}")
