import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d

# Setting up the default PMTs
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

# Plot each ring individually
for height in heights:
    ax.scatter(xs,ys,height,depthshade=0,color="grey",alpha=0.2,s=0.5)

# Vertex information
vtx = [0,0,0]

# Direction
vtx_dir = [1,0,0]

# Energy (testing)
bse = 50 # MeV

def dot(v1, v2):
    return sum([a*b for a,b in zip(v1,v2)])

def length(v):
    return math.sqrt(dot(v,v))

def angle(v1,v2):
    return math.acos(dot(v1,v2) / (length(v1) * length(v2)))

vect_scale = bse/5*math.sqrt(dot(vtx_dir,vtx_dir))

vtx_arrow = [vect_scale*a for a in vtx_dir]

# Cone (half) angle
thet_ch = math.pi/2

ax.quiver(*vtx,*vtx_arrow)

plt.show()