import numpy as np
from get_rdkit_3d_descriptors import *
import umap, pickle

def load_descriptors_from_npz(get_descriptors_from, downsample_rate=0.001, save_to=None):
    descriptors = []
    for block in get_descriptors_from.glob('*.npy'):
        block = np.load(block, allow_pickle=True)
        idx = np.arange(len(block))
        if downsample_rate != 1.0:
            #shuffle and take first N*downsample_rate
            np.random.shuffle(idx)
            idx = idx[:int(len(block)*downsample_rate)]
        descriptors.append(block[idx])

    descriptors = np.concatenate(descriptors)
    print('after downsampling, %d descriptors will be used to train the UMAP mapper' % len(descriptors))
    if save_to is not None:
        to_npz = {'descriptors': descriptors}
        np.savez(save_to, **to_npz)
    return descriptors


def get_mapper(descriptors, save_mapper_to=None):
      
    mapper = umap.UMAP(random_state=42).fit(descriptors)    
    if save_mapper_to:
        with open(save_mapper_to, 'wb') as f:
            pickle.dump(mapper, f)

    return mapper