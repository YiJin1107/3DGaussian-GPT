import numpy as np
n,m = 36,4
v = np.arange(108).reshape(36,3)
f = np.arange(12).reshape(4,3)
t = v[f,:]
print('-'*20)
print(t.shape)
print(t)
print(t.reshape(-1,9))