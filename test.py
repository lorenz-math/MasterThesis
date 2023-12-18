import numpy as np
import matplotlib.pyplot as plt
spacing = np.linspace(-0.5,0.5,5,endpoint=True)

def z(input):
    x = input[:,0]
    y = input[:,1]
    return x+y

#print(spacing)
#print(spacing.size)
x = np.repeat(spacing, spacing.size)
y = np.tile(np.flip(spacing), spacing.size)
XY=np.vstack((x,y)).T

Z2 = z(XY).reshape(spacing.size,spacing.size).T
print(Z2)

c = plt.imshow(Z2, cmap =plt.cm.RdBu, extent = [x.min(), x.max(), x.min(), x.max()]) 
plt.colorbar(c) 
plt.show()