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

def dot(v1, v2):
    return sum([a*b for a,b in zip(v1,v2)])

def length(v):
    return math.sqrt(dot(v,v))

def angle(v1,v2):
    return math.acos(dot(v1,v2) / (length(v1) * length(v2)))

# Vertex information
vtx = [0,0,0]

# Direction
vtx_dir = [1,0,0]
vtx_angle = angle(vtx_dir,[1,0,0])

# Line equation to find interaction of vtx with edge of detector
m_xy = math.tan(vtx_angle)
c_xy = vtx[1] - m_xy*vtx[0]

# Sub in y=mx+c to eqn for a circle to find x intersection with wall
# gets quadratic with these coeffs
a_1 = (m_xy*m_xy+1)
a_2 = 2*m_xy*c_xy
a_3 = c_xy*c_xy - hk_r*hk_r 

x_1 = -a_2+math.sqrt(a_2*a_2 - 4*a_1*a_3)
x_1 /= 2*a_1
y_1 = m_xy*x_1 + c_xy
x_2 = -a_2-math.sqrt(a_2*a_2 - 4*a_1*a_3)
x_2 /= 2*a_1
y_2 = m_xy*x_2 + c_xy

# For height, find h in 2D plane along vector
vtx_wall_xy = [x_1-vtx[0],y_1-vtx[1],0]
thet_h_wall = angle(vtx_wall_xy,vtx_dir)
h_1 = length(vtx_wall_xy)*math.tan(thet_h_wall)

ax.plot([vtx[0],x_1],[vtx[1],x_1],[vtx[0],h_1])

# Energy (testing)
bse = 50 # MeV

vect_scale = bse/5*math.sqrt(dot(vtx_dir,vtx_dir))

vtx_arrow = [vect_scale*a for a in vtx_dir]

# Cone (half) angle
thet_ch = math.pi/2

ax.quiver(*vtx,*vtx_arrow)

plt.show()