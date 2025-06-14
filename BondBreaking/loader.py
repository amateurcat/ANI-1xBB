import re, io, ase.io.sdf
from utils import IterableAdapter
import numpy as np


def picked_loader(dataset, index):

    def loader():
        with open(dataset,'r') as fr:
            l = fr.read().split("$$$$\n")   
            del l[-1]
            l = np.array(l)
        for s in l[index]:
            m,b = s.split("> <UniqueBond>")
            mol = ase.io.sdf.read_sdf(io.StringIO(m))
            b = (int(b.split(' ')[0]) +1,int(b.split(' ')[1]) +1)   #atom number in sdf start at 0, MUST add 1 here!
            name = "m%d"%(int(s.split('\n')[0]))
            yield (name, mol, b)   

    return IterableAdapter( lambda: loader() ) 