import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

def contour(f, x, y, level = np.logspace(0, 5, 35)):
	ax.contour(x, y, f(x,y), levels=level, norm=LogNorm(), cmap=plt.cm.jet)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

def contour_with_quiver(f, x, y, grad_x, grad_y, norm=LogNorm(), level = np.logspace(0, 5, 35),
	minima=None):
	dz_dx = grad_x(x,y)
	dz_dy = grad_y(x,y)

	fig, ax = plt.subplots()

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	if minima is not None:
		ax.plot(*minima, '*r', markersize=18)
	ax.quiver(x, y, -dz_dx, -dz_dy, alpha=.5)
	ax.set_xlabel('$a$')
	ax.set_ylabel('$b$')
	plt.axis('equal')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()

def surf(f, x, y, norm=LogNorm(), minima=None):
	ax = plt.axes(projection='3d', elev=50, azim=-50)

	ax.plot_surface(x, y, f(x,y), norm=norm, rstride=1, cstride=1,
	                edgecolor='none', alpha=.8, cmap=plt.cm.jet)

	if minima is not None:
		ax.plot(*minima, f(*minima), '*r', markersize=15)

	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.set_zlabel('$z$')

	plt.show()

def contour_with_path(f, x, y, paths, norm=LogNorm(), level=np.logspace(0, 5, 35), minima=None):
	fig, ax = plt.subplots()

	ax.contour(x, y, f(x,y), levels=level, norm=norm, cmap=plt.cm.jet)
	ax.quiver(paths[0,:-1], paths[1,:-1], paths[0,1:]-paths[0,:-1], paths[1,1:]-paths[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
	if minima is not None:
		ax.plot(*minima, 'r*', markersize=18)

	ax.set_xlabel('$a$')
	ax.set_ylabel('$b$')
	plt.axis('equal')

	# ax.set_xlim((xmin, xmax))
	# ax.set_ylim((ymin, ymax))

	plt.show()