import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import mesh


def render_mesh(obj, filename, resolution):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for vertices, indices in mesh.triangular_mesh(obj, resolution):
        print(vertices.shape)
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=indices)

    plt.show()
