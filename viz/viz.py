from numpy.core.fromnumeric import transpose
import pyglet
from pyglet.gl import glEnable, GL_DEPTH_TEST, GL_CULL_FACE
import argparse
import numpy as np
import math
import tables
import h5py
import os
import numba as nb

window = pyglet.window.Window(width=720, height=480, resizable=True)
window.projection = pyglet.math.Mat4().perspective_projection(0, 720, 0, 480, z_near=0.1, z_far=255)

label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

MAX_COLOUR = 2000
WORLD_SCALE = 400

@window.event
def on_draw():
    window.clear()
    #pyglet.graphics.draw(int(len(triangles) / 3), pyglet.gl.GL_TRIANGLES,
    #                     ('v3f', triangles))
    
    triangles.draw(pyglet.gl.GL_TRIANGLES)
    #label.draw()


def load_fits(path, dtype=np.float32):
    """
    Load a FITS image file, converting it to a Tensor
    Parameters
    ----------
    path : str
        The path to the FITS file.

    Returns
    -------
    torch.Tensor
    """
    from astropy.io import fits
    hdul = fits.open(path)
    data = np.array(hdul[0].data).astype(dtype)
    MAX_COLOUR = np.amax(data)
    return data


def size_image(data: np.ndarray, rez=800):
    """ 
    Size the image, finding the longest dimension and passing.
    Rez must be bigger than the image (for now anyway).
    Parameters
    ----------
    data : np.ndarray
        The loaded fits data

    rez : int
        The number of voxels on each dimension (default : 400)

    Returns
    -------
    np.ndarray
    """

    max_d = 0
    max_i = 0

    for i, d in enumerate(data.shape):
        if d > max_d:
            max_d = d
            max_i = i

    assert(max_d < rez)

    padding = []
    for i, d in enumerate(data.shape):
        pl = int(math.floor((rez - d) / 2))
        pr = int(math.ceil((rez - d) / 2))
        padding.append((pl, pr))

    os.remove("./cache.hdf5")
    hdf5_store = h5py.File("./cache.hdf5", "a")
    padded = np.pad(data, padding)
    #cx = hdf5_store.create_dataset("data", padded.shape, compression="gzip", data=padded)

    return padded


@nb.jit(nopython=True, parallel=True)
def get_threshold(data, threshold):
    sim_vec = []

    for m in range(data.shape[0]):
        for n in range(data.shape[1]):
            for p in range(data.shape[2]):
                if data[m, n, p] >= threshold:
                    sim_vec.append((m, n, p))

    return sim_vec


@nb.jit(nopython=True, parallel=False)
def interp_edge(vertvals, e, x, y, z, cutoff):
    # First Vertex Positions
    v0 = [0.0, 0.0, 0.0]
    vid0 = tables.edge_to_verts[e * 2]
    v0[0] = tables.vert_to_cart[vid0 * 3]
    v0[1] = tables.vert_to_cart[vid0 * 3 + 1]
    v0[2] = tables.vert_to_cart[vid0 * 3 + 2]

    # Second vertex Position
    v1 = [0.0, 0.0, 0.0]
    vid1 = tables.edge_to_verts[e * 2 + 1]
    v1[0] = tables.vert_to_cart[vid1 * 3]
    v1[1] = tables.vert_to_cart[vid1 * 3 + 1]
    v1[2] = tables.vert_to_cart[vid1 * 3 + 2]

    diff = math.fabs(vertvals[vid0] - vertvals[vid1])
    assert(diff != 0)
    vr = math.fabs(vertvals[0] - cutoff) / diff

    vf = v0
    vf[0] += (x + v1[0] * vr) * WORLD_SCALE - 1.0
    vf[1] += (y + v1[1] * vr) * WORLD_SCALE - 1.0
    vf[2] += (z + v1[2] * vr) * WORLD_SCALE - 1.0

    # TODO For now, just equally split the colour. Eventually properly interpolate
    c = vertvals[vid0] / MAX_COLOUR * 255
    return (vf, (c, c, c, 0.5))


# Loop over the indices and find the cubes. Return list of tris in x,y,z verts 0 to 2
@nb.jit(nopython=True, parallel=False)
def cuuubes(data, cutoff=200.0):
    indices = get_threshold(data, cutoff)
    print("Num Voxels to Check", len(indices))
    test_cubes = []
    final_tris = []
    final_colours = []

    # TODO - potential speedup if we choose the smallest of either < or > than threshold

    # Loop through and make some cubes
    for z, y, x in indices:
        # Sample each point and combine to a byte
        v0 = data[z][y][x]
        v1 = data[z + 1][y][x]
        v2 = data[z + 1][y][x + 1]
        v3 = data[z][y][x + 1]
        v4 = data[z][y + 1][x]
        v5 = data[z + 1][y + 1][x]
        v6 = data[z + 1][y + 1][x + 1]
        v7 = data[z][y + 1][x + 1]

        c0 = int(v0 >= cutoff)
        c1 = int(v1 >= cutoff)
        c2 = int(v2 >= cutoff)
        c3 = int(v3 >= cutoff)
        c4 = int(v4 >= cutoff)
        c5 = int(v5 >= cutoff)
        c6 = int(v6 >= cutoff)
        c7 = int(v7 >= cutoff)
        lookup = c7 << 7 | c6 << 6 | c5 << 5 | c4 << 4 | c3 << 3 | c2 << 2 | c1 << 1 | c0

        if lookup != 0 and lookup != 255:
            test_cubes.append(
                ([z, y, x], lookup, [v0, v1, v2, v3, v4, v5, v6, v7]))

    print("Test Cubes complete.")

    for tcube in test_cubes:
        lookup = tcube[1]
        n_polys = tables.case_to_numpolys[lookup]
        p_start = 15 * lookup
        z, y, x = tcube[0]
        verts = tcube[2]

        for i in range(n_polys):
            e0 = tables.tri_table[p_start + i * 3]
            e1 = tables.tri_table[p_start + i * 3 + 1]
            e2 = tables.tri_table[p_start + i * 3 + 2]
            assert(e0 != -1 and e1 != -1 and e2 != -1)

            # interpolate all 3 edges
            v0, c0 = interp_edge(verts, e0, x, y, z, cutoff)
            v1, c1 = interp_edge(verts, e1, x, y, z, cutoff)
            v2, c2 = interp_edge(verts, e2, x, y, z, cutoff)

            # Flat vertex list
            for i in range(3):
                final_tris.append(v0[i])
            for i in range(4):
                final_colours.append(c0[i])
            
            for i in range(3):
                final_tris.append(v1[i])

            for i in range(4):
                final_colours.append(c1[i])

            for i in range(3):
                final_tris.append(v2[i])

            for i in range(4):
                final_colours.append(c2[i])

    print("Triangles computed.")
    return final_tris, final_colours


if __name__ == "__main__":
    global triangles

    parser = argparse.ArgumentParser(
        description='Pyglet based visualisation for our 3D fits images.')
    parser.add_argument('--image', default="../test/images/raw.fits",
                        help='Path to a 3D fits image.')
    parser.add_argument("--rez", type=int, default=800,
                        help="The resolution along each axis.")
    parser.add_argument('--cutoff', type=float, default=200.0, metavar='LR',
                        help='The cutoff value for our cuuuubes.')

    args = parser.parse_args()
    WORLD_SCALE = 2.0 / args.rez
    data = load_fits(args.image)
    data = size_image(data, args.rez)
    tris, colours = cuuubes(data, args.cutoff)
    triangles = pyglet.graphics.vertex_list(int(len(tris) / 3), ('position3f', tris), ('colors4f', colours))

    print(tris[0:3])

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    pyglet.app.run()
