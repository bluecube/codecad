import matplotlib.pyplot as plt

from . import mesh


def render_mesh(obj, _filename=None):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    for vertices, indices in mesh.triangular_mesh(obj):
        ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=indices
        )

    plt.show()
