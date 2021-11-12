
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
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face " + str(int(len(vertices) / 9)) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertex list
        for vidx in range(0, len(vertices), 3):
            v = vertices[vidx]
            f.write(str(round(v, 4)) + " ")
            v = vertices[vidx + 1]
            f.write(str(round(v, 4)) + " ")
            v = vertices[vidx + 2]
            f.write(str(round(v, 4)) + " ")
            c = int(colours[vidx])
            f.write(str(c) + " ")
            c = int(colours[vidx + 1])
            f.write(str(c) + " ")
            c = int(colours[vidx + 2])
            f.write(str(c) + "\n")

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

    #os.remove("./cache.hdf5")
    #hdf5_store = h5py.File("./cache.hdf5", "a")
    padded = np.pad(data, padding)
    #cx = hdf5_store.create_dataset("data", padded.shape, compression="gzip", data=padded)

    return padded


@nb.jit(nopython=True, parallel=True)
def get_threshold(data, threshold):
    sim_vec = []

    for z in range(data.shape[0] - 1):
        for y in range(data.shape[1] - 1):
            for x in range(data.shape[2] - 1):
                v0 = data[z][y][x]
                v1 = data[z + 1][y][x]
                v2 = data[z + 1][y][x + 1]
                v3 = data[z][y][x + 1]
                v4 = data[z][y + 1][x]
                v5 = data[z + 1][y + 1][x]
                v6 = data[z + 1][y + 1][x + 1]
                v7 = data[z][y + 1][x + 1]

                c0 = int(v0 >= threshold)
                c1 = int(v1 >= threshold)
                c2 = int(v2 >= threshold)
                c3 = int(v3 >= threshold)
                c4 = int(v4 >= threshold)
                c5 = int(v5 >= threshold)
                c6 = int(v6 >= threshold)
                c7 = int(v7 >= threshold)
                lookup = c7 << 7 | c6 << 6 | c5 << 5 | c4 << 4 | c3 << 3 | c2 << 2 | c1 << 1 | c0

                if lookup != 0 and lookup != 255:
                    sim_vec.append(([z, y, x], lookup,  [v0, v1, v2, v3, v4, v5, v6, v7]))

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
    vf[1] += dp[1] * vr + y
    vf[2] += dp[2] * vr + z

    vf[0] = (vf[0] * WORLD_SCALE) - 1.0
    vf[1] = (vf[1] * WORLD_SCALE) - 1.0
    vf[2] = (vf[2] * WORLD_SCALE) - 1.0

    c = (vertvals[vid0] * (1.0 - vr) + vertvals[vid1] * vr) / MAX_COLOUR * 255
    return (vf, (c, c, c))


# Loop over the indices and find the cubes. Return list of tris in x,y,z verts 0 to 2
@nb.jit(nopython=True, parallel=False)
def cuuubes(data, cutoff=200.0):
    test_cubes = get_threshold(data, cutoff)
   
    print("Test Cubes complete.")
    final_tris = []
    final_colours = []

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

            # We invert Z as the FITS Z starts from the top of the stack down
            v0[2] = v0[2] * -1
            v1[2] = v1[2] * -1
            v2[2] = v2[2] * -1

            # Flat vertex list
            for i in range(3):
                final_tris.append(v0[i])

            for i in range(3):
                final_colours.append(c0[i])
            
            for i in range(3):
                final_tris.append(v1[i])

            for i in range(3):
                final_colours.append(c1[i])

            for i in range(3):
                final_tris.append(v2[i])

            for i in range(3):
                final_colours.append(c2[i])

    print("Triangles computed.")
    return final_tris, final_colours


if __name__ == "__main__":
    global triangles

    parser = argparse.ArgumentParser(
        description='Pyglet based visualisation for our 3D fits images.')
    parser.add_argument('--image', default="../test/images/raw.fits",
                        help='Path to a 3D fits image.')
    parser.add_argument('--savepath', default="vis.ply",
                        help='Path to a 3D PLY file we are saving')
    parser.add_argument("--rez", type=int, default=800,
                        help="The resolution along each axis.")
    parser.add_argument('--cutoff', type=float, default=400.0,
                        help='The cutoff value for our cuuuubes.')

    args = parser.parse_args()
    WORLD_SCALE = 2.0 / args.rez
    data = load_fits(args.image)
    data = size_image(data, args.rez)
    tris, colours = cuuubes(data, args.cutoff)
    save_ply(args.savepath, tris, colours)
