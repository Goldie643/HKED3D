import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# Setting up the default PMTs
n_rings = 100
n_per_ring = 400

hk_r = 20
hk_hh = 20

thetas = np.linspace(0,2*np.pi,n_per_ring)
heights = np.linspace(-hk_hh,hk_hh,n_rings)

xs = [hk_r*np.cos(theta) for theta in thetas]
ys = [hk_r*np.sin(theta) for theta in thetas]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-hk_r,hk_r)
ax.set_ylim(-hk_r,hk_r)
ax.set_zlim(-hk_hh,hk_hh)

# Plot each ring individually
for height in heights:
    ax.scatter(xs,ys,height,depthshade=0,color="grey",alpha=0.2,s=0.5)

def length(v):
    return np.sqrt(np.dot(v,v))

def angle(v1,v2):
    return np.acos(np.dot(v1,v2) / (length(v1) * length(v2)))

# Vertex information
vtx = np.array([0,0,0])

# Direction
vtx_dir = np.array([1,0,0])
cone_scale = 5 
cone_spine = cone_scale*vtx_dir
# vtx_angle = angle(vtx_dir,[1,0,0])

# Make a new vector not equal to vtx_dir
not_cone_spine = np.array([1,0,0])
if (vtx_dir == not_cone_spine).all():
    not_cone_spine = np.array([0,1,0])

# Make normalised vector perpendicular to vtx_dir
n1 = np.cross(cone_spine,not_cone_spine)
n1 = n1/length(n1)
n2 = np.cross(cone_spine,n1)
n2 = n2/length(n2)

# Angles to make points on the cone edge for
theta = np.linspace(0, 2 * np.pi, 50)
R = 10
# Generate coordinates for surface
cone_face = [vtx[i] + cone_spine[i] * cone_scale + R *
            np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

# cone = [np.append(cone_pt,x) for cone_pt,x in zip(cone_face,vtx)]

# For tri surface, need to define triangles of vtx,cone_face_i,cone_face_i+1
cone = []

# Go through each dimension
for i,axis in enumerate(cone_face):
    # Each value in each dimension
    axis_values = []
    for j,x in enumerate(axis):
        axis_values.append(vtx[i])
        axis_values.append(x)
    cone.append(axis_values)

# ax.plot_trisurf(cone_xs,cone_ys,cone_zs)
# ax.plot_trisurf(*cone)
ax.plot(*cone)

# # Line equation to find interaction of vtx with edge of detector
# m_xy = math.tan(vtx_angle)
# c_xy = vtx[1] - m_xy*vtx[0]

# # Sub in y=mx+c to eqn for a circle to find x intersection with wall
# # gets quadratic with these coeffs
# a_1 = (m_xy*m_xy+1)
# a_2 = 2*m_xy*c_xy
# a_3 = c_xy*c_xy - hk_r*hk_r 

# x_1 = -a_2+math.sqrt(a_2*a_2 - 4*a_1*a_3)
# x_1 /= 2*a_1
# y_1 = m_xy*x_1 + c_xy
# x_2 = -a_2-math.sqrt(a_2*a_2 - 4*a_1*a_3)
# x_2 /= 2*a_1
# y_2 = m_xy*x_2 + c_xy

# # For height, find h in 2D plane along vector
# vtx_wall_xy = [x_1-vtx[0],y_1-vtx[1],0]
# thet_h_wall = angle(vtx_wall_xy,vtx_dir)
# h_1 = length(vtx_wall_xy)*math.tan(thet_h_wall)

# ax.plot([vtx[0],x_1],[vtx[1],x_1],[vtx[0],h_1])


# Energy (testing)
bse = 50 # MeV

vect_scale = bse/5*np.sqrt(np.dot(vtx_dir,vtx_dir))

vtx_arrow = [vect_scale*a for a in vtx_dir]

# Cone (half) angle
thet_ch = np.pi/2

ax.quiver(*vtx,*vtx_arrow)

plt.show()