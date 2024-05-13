import numpy as np

arr = np.random.randn(5000, 5000)
np.save("dummy.npy", arr)