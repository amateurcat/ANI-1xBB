import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors3D
import sys, io

sys.path.append('../../')
from converters import *

def get_3d_descriptors(n,c):
    pdb_io = io.StringIO()
    std_to_pdb(n, c, pdb_io)
    pdb_io.seek(0)
    mol = Chem.MolFromPDBBlock(pdb_io.getvalue(), sanitize=False)
    ret = np.array(list(Descriptors3D.CalcMolDescriptors3D(mol).values()))
    return ret

def block_3d_descriptors(db, save_to=None):

    if isinstance(db, Path):
        db = np.load(db, allow_pickle=True)
        numbers, coord = db['numbers'], db['coord']
    elif isinstance(db, dict):
        numbers, coord = db['numbers'], db['coord']
    elif isinstance(db, tuple):
        numbers, coord = db

    ret = []
    for n, c in zip(numbers, coord):
        ret.append(get_3d_descriptors(n,c))

    ret = np.array(ret)
    if save_to is not None:
        np.save(save_to, ret)

    return ret

def get_bb_descriptors_and_fods(bb_dir=Path('/storage/users/shuhao/bb_datasets_final/bond_breaking_4el'), 
                                saved_descriptors_dir=Path('/storage/users/shuhao/descriptors'), 
                                downsample_rate=0.01, 
                                save_to='./selected_bb4el_descriptors_and_fods.npz'):
    
    descriptors = []
    fods = []
    for i in range(3, 24):
        db = bb_dir/ (f"{i:03}.npz")
        desc_db = saved_descriptors_dir/('bb4el_'+db.stem+'_rdkit_3d_descriptors.npy')

        block = np.load(desc_db, allow_pickle=True)
        idx = np.arange(len(block))
        if downsample_rate != 1.0:
            #shuffle and take first N*downsample_rate
            np.random.shuffle(idx)
            idx = idx[:int(len(block)*downsample_rate)]
        descriptors.append(block[idx])

        db = np.load(db, allow_pickle=True)
        fods.append(db['b973c_etemp5000.fod'][idx])

    descriptors = np.concatenate(descriptors)
    fods = np.concatenate(fods)

    to_npz = {'descriptors': descriptors, 'b973c_etemp5000.fod': fods}
    np.savez(save_to, **to_npz)

if __name__ == '__main__':
    save_to_dir = Path('/storage/users/shuhao/descriptors')
    #ANI1x_B973c_dir = Path('/storage/users/shuhao/ani1x_b973c_07152021')
    bb_dir = Path('/storage/users/shuhao/bb_datasets_final/bond_breaking_4el')

    for i in range(3, 24):
        db = bb_dir/ (f"{i:03}.npz")
        print(db.stem)
        save_to = db.stem+'_3d_descriptors.npy'
        block_3d_descriptors(db, save_to=save_to_dir/('bb4el_'+db.stem+'_rdkit_3d_descriptors.npy'))

