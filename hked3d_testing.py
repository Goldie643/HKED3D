import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d

n_rings = 100
n_per_ring = 400

hk_r = 20
hk_hh = 20

thetas = np.linspace(0,2*math.pi,n_per_ring)
heights = np.linspace(-hk_hh,hk_hh,n_rings)

xs = [hk_r*np.cos(theta) for theta in thetas]
ys = [hk_r*np.sin(theta) for theta in thetas]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for height in heights:
    ax.scatter(xs,ys,height,depthshade=0,color="grey",alpha=0.2,s=0.5)

hs = [hk_hh]*n_per_ring

xs = xs[:len(xs)]
ys = ys[:len(ys)]
hs = hs[:len(hs)]

xs.append(0)
ys.append(0)
hs.append(0)

ax.plot_trisurf(xs,ys,hs)

plt.show()