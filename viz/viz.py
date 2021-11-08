
import argparse
import numpy as np
import math
import tables
import h5py
import os
import numba as nb

MAX_COLOUR = 2000
WORLD_SCALE = 400


def save_ply(path, vertices, colours):
    """
    Save a basic ascii ply file that has triangles and colour.

    Parameters
    ----------
    path : str
        A path and filename for the save file
    vertices : list
        A list of vertices, every 3 is a triangle. A flat
        triangle list.
    colours : list
        A list of RGB colours, 255.

    Returns
    -------
    None

    """

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(len(vertices) / 3) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face " + str(int(len(vertices) / 9)) + "\n")
        #f.write("element face " + str(0) + "\n")

        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertex list
        for vidx in range(0, len(vertices), 3):
            v = vertices[vidx]
            f.write(str(round(v, 4)) + " ")
            v = vertices[vidx + 1]
            f.write(str(round(v, 4)) + " ")
            v = vertices[vidx + 2]
            f.write(str(round(v, 4)) + "\n")
        
        # Face list
        for vidx in range(0, int(len(vertices) / 3), 3):
            f.write("3 " + str(vidx) + " " + str(vidx + 1) + " " + str(vidx + 2) + "\n")


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
    global MAX_COLOUR
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
                if data[m, n, p] >= 300:
                    sim_vec.append((m, n, p))

    return sim_vec


@nb.jit(nopython=True, parallel=False)
def interp_edge(vertvals, e, x, y, z, cutoff):
    global MAX_COLOUR

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
    vr = math.fabs(vertvals[vid0] - cutoff) / diff
    assert(vr <= 1.0 and vr >= 0.0)
    dp = v1
    dp[0] -= v0[0]
    dp[1] -= v0[1]
    dp[2] -= v0[2]

    vf = v0
 
    vf[0] += dp[0] * vr + x
    vf[1] += dp[1] * vr + y # Remember, Z is up in this system
    vf[2] += dp[2] * vr + z

    vf[0] *= WORLD_SCALE - 1.0
    vf[1] *= WORLD_SCALE - 1.0
    vf[2] *= WORLD_SCALE - 1.0

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
    parser.add_argument('--cutoff', type=float, default=600.0,
                        help='The cutoff value for our cuuuubes.')

    args = parser.parse_args()
    WORLD_SCALE = 2.0 / args.rez
    data = load_fits(args.image)
    data = size_image(data, args.rez)
    tris, colours = cuuubes(data, args.cutoff)
   
    save_ply("vis.ply", tris, colours)
