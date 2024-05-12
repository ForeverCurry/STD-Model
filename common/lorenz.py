# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Lorenz paramters and initial conditions
# sigma, beta, rho = 10, 2.667, 28
# u0, v0, w0 = 0, 1, 1.05

# # Maximum time point and total number of time points
# tmax, n = 100, 10000

# def lorenz(X, t, sigma, beta, rho):
#     """The Lorenz equations."""
#     u, v, w = X
#     up = -sigma*(u - v)
#     vp = rho*u - v - u*w
#     wp = -beta*w + u*v
#     return up, vp, wp

# # Integrate the Lorenz equations on the time grid t
# t = np.linspace(0, tmax, n)
# f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
# x, y, z = f.T

# # Plot the Lorenz attractor using a Matplotlib 3D projection
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111,projection='3d')

# # Make the line multi-coloured by plotting it in segments of length s which
# # change in colour across the whole time series.
# s = 10
# c = np.linspace(0,1,n)
# for i in range(0,n-s,s):
#     ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color='black',alpha=0.78)

# # Remove all the axis clutter, leaving just the curve.
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()
# fig.savefig(r'D:\ML\Time_series\mymodel\png\lorenz.png',transparent=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
# Create a 3D array
# meshgrid creates a rectangular grid out of an array of x values and an array of y values.
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
# x, y = np.meshgrid(x, y)

# # A 3D sinusoidal manifold
# z = np.sin(np.sqrt(x**2 + y**2))

# # Create a figure and a 3D Axes
# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # Plot the surface.
# ax.plot_surface(x, y, z, cmap='viridis')

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(plt.LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# # Add a color bar which maps values to colors.
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()
# fig.savefig(r'D:\ML\Time_series\mymodel\png\project.png',transparent=True)


# Create a figure and axis
# Define the four points (x, y, z)
points = np.array([[3, 2.3], [5.0, 2.4], [8, 2.15], [12, 2.45]])
points2 = np.array([[0., 2.], [4, 2.5], [15, 2.5], [11, 2.]])
# Create a figure
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)

# Plot the points
ax.scatter(points[:, 0], points[:, 1], marker='.',color='black',s=550)

# Interpolate a B-spline curve
tck, u = splprep(points.T, u=None,k=2, s=0.0)
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck)

# Plot the curve
ax.plot(x_new, y_new, 'k--',linewidth=2)
# arrow_start = [x[len(x)//2], y[len(y)//2]]
# arrow_end = [x[len(x)//2 + 1], y[len(y)//2 + 1]]
# ax.annotate('', xy=arrow_end, xytext=arrow_start,
#              arrowprops=dict(facecolor='black', shrink=0.05))
# Add a diamond plane
polygon = Polygon(points2, closed=True)
p = PatchCollection([polygon], alpha=0.5)
p.set_color((73/255, 149/255, 198/255))
ax.add_collection(p)

# Add a text label
for i, txt in enumerate([f'$\\mathbf{{a}}_0$', f'$\\mathbf{{a}}_1$', f'$\\mathbf{{a}}_2$', f'$\\mathbf{{a}}_{{l-1}}$']):
    if i==0:
        ax.annotate(txt, (points[i, 0]+0.3, points[i, 1]-0.05),fontsize=28,color='black')
    else:
        ax.annotate(txt, (points[i, 0]+0.1, points[i, 1]-0.1),fontsize=28,color='red')

# Add the text F in the upper right corner of the diamond
ax.text(np.min(points[:, 0])-1, np.min(points[:, 1])-0.1, f'$\\mathbb{{R}}^k$', color='black',fontsize=48)
ax.text(6.3, 2.15, f'$G$', color='red',fontsize=28)
ax.set_xlim(0.,15)
ax.set_ylim(1.5,3.5)
# Show the plot
ax.set_axis_off()
plt.show()
plt.tight_layout()
fig.savefig(r'D:\ML\Time_series\mymodel\png\rep_dyn.png',transparent=True)