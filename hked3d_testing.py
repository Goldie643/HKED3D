import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# Setting up the default PMTs
n_rings = 100
n_per_ring = 400

hk_r = 20
hk_hh = 20


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-hk_r,hk_r)
ax.set_ylim(-hk_r,hk_r)
ax.set_zlim(-hk_hh,hk_hh)

def length(v):
    return np.sqrt(np.dot(v,v))

def angle(v1,v2):
    return np.acos(np.dot(v1,v2) / (length(v1) * length(v2)))

def generate_cone(vtx,vtx_dir,theta_ch):
    cone_scale = 20 
    cone_spine = cone_scale*vtx_dir
    cone_r = cone_scale*np.tan(np.deg2rad(thet_ch))
    # Make a new vector not equal to vtx_dir
    not_cone_spine = np.array([1,0,0])
    if (vtx_dir == not_cone_spine).all():
        not_cone_spine = np.array([0,1,0])

    # Make normalised vector perpendicular to vtx_dir
    n1 = np.cross(vtx_dir,not_cone_spine)
    n1 = n1/length(n1)
    n2 = np.cross(vtx_dir,n1)
    # n2 = n2/length(n2)

    # Angles to make points on the cone edge for
    n = 100
    t = np.linspace(0,cone_scale,n)
    theta = np.linspace(0,2 * np.pi,n)
    t, theta = np.meshgrid(t, theta)
    R = np.linspace(0,cone_r,n)
    # Generate coordinates for surface
    cone = [vtx[i] + vtx_dir[i] * t + R *
                np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    return cone

# Vertex information
vtx = np.array([5,9,-4])
vtx_dir = np.array([-1,-2,2])
vtx_dir = vtx_dir/length(vtx_dir)
thet_ch = 30 # deg
cone = generate_cone(vtx, vtx_dir,thet_ch)
ax.plot_surface(*cone,alpha=0.5,color="dodgerblue")

# Colour PMTs based on how far from points in the cone they are (slow)
ring_thetas = np.linspace(0,2*np.pi,n_per_ring)
ring_heights = np.linspace(-hk_hh,hk_hh,n_rings)

ring_xs = [hk_r*np.cos(theta) for theta in ring_thetas]
ring_ys = [hk_r*np.sin(theta) for theta in ring_thetas]

pmt_xs =[]
pmt_ys =[]
pmt_zs =[]
# Add each ring
for height in ring_heights:
    pmt_xs.extend(ring_xs)
    pmt_ys.extend(ring_ys)
    pmt_zs.extend([height]*n_per_ring)

# Get list of vertexes
pmt_vtxs = [[x,y,z] for x,y,z in zip(pmt_xs,pmt_ys,pmt_zs)]

cone = [a.ravel() for a in cone]
cone_vtxs = [[x,y,z] for x,y,z in zip(*cone)]

# pmt_min_dxs =[]
# for pmt_vtx in pmt_vtxs:
#     distances = []
#     for cone_vtx in cone_vtxs:
#         dx = cone_vtx[0] - pmt_vtx[0]
#         dy = cone_vtx[1] - pmt_vtx[1]
#         dz = cone_vtx[2] - pmt_vtx[2]
#         distances.append(np.sqrt(dx*dx+dy*dy+dz*dz))
#     pmt_min_dxs.append(min(distances))

# print(min(pmt_min_dxs))
# print(max(pmt_min_dxs))

ax.scatter(pmt_xs,pmt_ys,pmt_zs,depthshade=0,color="grey",alpha=0.2,s=0.5)

# vtx_angle = angle(vtx_dir,[1,0,0])

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