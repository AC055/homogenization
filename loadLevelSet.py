import numpy as np
import matplotlib.pyplot as plt
import meshio

plt.close('all')

# If needed, reinit can be turned to True
reinit = False


# Load levelset
lsLoad = np.loadtxt('stellite513.ls')
# lsLoad = np.loadtxt('composite513.ls')
nx = int(lsLoad[0])
ny = int(lsLoad[1])
stepx = int(lsLoad[2])
stepy = int(lsLoad[3])
lsLoad_ = lsLoad[6:].reshape((nx,ny))


# Blur a little
from scipy.ndimage import gaussian_filter
lsLoad = gaussian_filter(lsLoad_, sigma=1.2)
# lsLoad = lsLoad_
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
# (if needed) Reinitialize as a distance function
if reinit:
    
    # Find approximate zero level set to preserve it
    zero_mask = np.abs(lsLoad) < 0.5 * stepx
    
    # Compute signed distance 
    inside = lsLoad < 0
    outside = lsLoad >= 0
    dist_inside = distance_transform_edt(inside) * stepx
    dist_outside = distance_transform_edt(outside) * stepx

    lsLoad_reinit = dist_outside - dist_inside
    
    # Correct the zero level set using linear interpolation
    # This preserves the interface location from the original level set
    for i in range(nx):
        for j in range(ny):
            if zero_mask[i, j]:
                # Keep original value near the interface for smoothness
                lsLoad_reinit[i, j] = lsLoad[i, j]
    
    # Smooth the reinitialized level set to remove artifacts
    from scipy.ndimage import gaussian_filter
    lsLoad = gaussian_filter(lsLoad_reinit, sigma=0.8)

# Create mesh
bbmin = np.array([0., 0.])
bbmax = np.array([(nx-1) * stepx, (ny-1) * stepy])
xx = np.linspace(0, bbmax[0] - bbmin[0], nx)
yy = np.linspace(0, bbmax[1] - bbmin[1], ny)
XX, YY = np.meshgrid(xx,yy)

pts = np.vstack((XX.flatten(), YY.flatten(), 0.*XX.flatten())).T
quads = np.zeros(((nx-1) * (ny-1), 4), dtype=int)
for jelt in range(ny-1):
    for ielt in range(nx-1):
        base = ielt + jelt * nx
        quads[ielt + jelt * (nx-1)] = [base, base+1, base+nx+1, base+nx]

# Convert quads to triangles
elts = np.vstack([quads[:,[0,1,2]], quads[:,[0,2,3]]])  
mesh = meshio.Mesh(pts, {'triangle': elts}, point_data={'ls': lsLoad.flatten()})
mesh.write('loadLevelSet_sate.msh', file_format='gmsh')
mesh.write('loadLevelSet_sate.vtu')

# Adapter function to evaluate the level set at arbitrary points:
# You could also locate the pixel in which the point is located and return the value of the level set at that pixel, but interpolation is more accurate.
def levelSetAdapter(xy):
    # Interpolate the level set function at given (x, y) coordinates
    interpolator = RegularGridInterpolator((xx, yy), lsLoad.T)
    return interpolator(xy)

 
# set the z-coordinate of the mesh points to the level set values
# mesh.points[:, 2] = levelSetAdapter(mesh.points[:, :2]) 
# print(mesh.points) 

# lsVals = levelSetAdapter(np.array([[0.5, 0.5], [1.0, 1.0]]))  # Example usage
# print(lsVals)

# nnn