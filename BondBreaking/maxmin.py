import numpy as np
from itertools import islice
from argparse import ArgumentParser
import os, random, io, ase.io
from ase.io.xyz import write_xyz


def bpdist(x):
    """ Batched triangular distance matrix
    """
    assert x.shape[2] == 3
    assert x.ndim == 3
    _n = x.shape[1]
    dmat = np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)
    mask = np.tril_indices(_n, -1)
    return dmat[:, mask[0], mask[1]]


def brdist(x, y):
    """ RMS of relative differences
    """
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[1]
    x = x[:, None, :]
    y = y[None, :, :]
    diff = (x - y) / (0.5 * (x + y))
    rms = (diff ** 2).mean(axis=-1) ** 0.5
    return rms


def maxmin(v, seed=0, mindist=0.0, yield_seed=True):
    if yield_seed:
        yield seed, 0
    n = v.shape[0]
    idx = np.arange(n)
    mask = np.zeros(n, dtype=np.bool)
    dmat = np.zeros((n, n), dtype=v.dtype)

    last_add = seed
    while sum(~mask) > 1:
        mask[last_add] = True
        dmat[last_add, ~mask] = brdist(v[last_add:last_add+1], v[~mask])
        mat = dmat[mask][:, ~mask]
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        a = np.arange(mat.shape[1])
        mi = np.argmin(mat, axis=0)
        ma = np.argmax(mat[mi, a])
        i = idx[~mask][ma]
        d = mat[mi, a][ma]
        if d > mindist:
            yield i, d
        else:
            break
        last_add = i


def guess_type(filename):
    assert '.' in filename
    return os.path.splitext(filename)[1][1:]


def getcoords(mols):
    return np.asarray([mol.positions for mol in mols])


def find_interest(traj_file, min_diff=0.05, nmax=10, seed=0):

    mol_name = traj_file.split("_total.xyz")[0]
    try:
        mols = ase.io.read(traj_file,':')
        #assert mols, f'Failed to find any conformers in {traj_file}'
        dmat = bpdist(getcoords(mols))

        if seed == -1:
            seed = random.choice(range(len(mols)))
        
        maxmin_result = io.StringIO()
        _n = 0
        for i, d in maxmin(dmat, seed=seed, mindist=min_diff, yield_seed=True):
            write_xyz(maxmin_result, mols[i])
            _n += 1
            if _n == nmax:
                break

        maxmin_result.seek(0)
        with open("%s_maxmined.xyz"%(mol_name), 'w') as f:
            f.write(maxmin_result.read())
    except:
        pass
            