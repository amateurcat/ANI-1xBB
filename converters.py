# Most of the general used converter functions in this project

import numpy as np
import ase.io
from ase import Atoms
from ase.io.trajectory import Trajectory
from pathlib import Path

PERIODIC_TABLE = """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()

def process_single_std(f):
    def t(numbers, coord, *args, **kwargs):
        if len(numbers.shape) == 1:
            numbers = np.atleast_2d(numbers)
        if len(coord.shape) == 2:
            coord = np.expand_dims(coord, axis=0)   ###!!! DO NOT use np.atleast_3d(c)
        return f(numbers, coord, *args, **kwargs)
    return t

def numbers2species(numbers, order=None):
    #convert atomic number to atomic symbol, by default use periodic table order
    #order is a list of atomic symbols in the desired order
    if order is None:
        ret = np.array([PERIODIC_TABLE[n-1] for n in numbers.flatten()]).reshape(*numbers.shape)
    else:
        ret = np.array([order[n] for n in numbers.flatten()]).reshape(*numbers.shape)
        
    return ret

def species2numbers(species, order=None):
    #Inverse function of the one above, by default use periodic table order
    #order is a list of atomic symbols in the desired order
    if order is None:
        ret = np.array([PERIODIC_TABLE.index(s)+1 for s in species.flatten()]).reshape(*species.shape)
    else:
        ret = np.array([order.index(s) for s in species.flatten()]).reshape(*species.shape)
        
    return ret


def convert_char_list(char_list):
    '''
    creating an ASE Atoms instance requires a sequence of chemical symbols as the element type input
    e.g. when creating a H2O molecule, you need sequence "HHO" and corresponding coordinates
    however, out dataset record elements by array of atomic numbers like np.array([1,1,8]) for H2O
    and the helper function numbers2species can only conver it into char array like np.array(['H','H','O'])
    so here is a function to make chemical symbols sequence
    '''
    s = ''
    for c in char_list:
        s += c
    return s
    
def to_Atoms(numbers,coord):
    '''
    create an ASE Atoms instance with given conformer in the standard format
    '''
    return Atoms(convert_char_list(numbers2species(numbers)), positions=coord)

@process_single_std
def std_to_xyz(numbers, coord, save_to):
    '''
    function to write an array of numbers and coord to a xyz file
    no matter how many conformers are in the input array
    they will be write into the same xyz file
    '''
    atoms_list = [to_Atoms(n,c) for n,c in zip(numbers, coord)]
    ase.io.write(save_to, atoms_list,'xyz')

@process_single_std
def std_to_pdb(numbers, coord, save_to):
    '''
    function to write an array of numbers and coord to a pdb file
    The output pdb file does not contain cell or connection information
    '''
    atoms_list = [to_Atoms(n,c) for n,c in zip(numbers, coord)]
    ase.io.write(save_to, atoms_list, format='proteindatabank')
    
def xyz_to_std(read_from):
    '''
    read an xyz file with 1 or more conformers
    return standard format (numbers, coord)
    '''
    r = ase.io.read(read_from,':')
    numbers = []
    coord= []
    for a in r:
        numbers.append(a.symbols.numbers)
        coord.append(a.positions)
    return np.stack(numbers), np.stack(coord)
    
def sxyz2xyzs(read_from, save_to=None):
    '''
    convert a xyz file with N conformers to N xyz files with single conformer
    '''
    p = Path(read_from)
    name = p.stem
    if save_to==None:
        save_to = p.parent
    else:
        save_to = Path(save_to)
        
    r = ase.io.read(read_from,':')
    for i,a in enumerate(r):
        ase.io.write(save_to/(name+'_'+str(i)), a,'xyz')
    
    
def to_Traj(numbers_array, coord_array, save_to):
    '''
    function to generate ASE trajectory file
    this is useful for MD in ASE, but cannot be visualize in most of softwares. 
    '''
    atoms_list = [to_Atoms(n,c) for n,c in zip(numbers_array, coord_array)]
    return Trajectory(save_to, 'w', atoms_list)
    
def traj_to_xyz(traj_file, save_to='test.xyz'):
    '''
    convert ASE trajectory file into xyz file
    '''
    t = ase.io.Trajectory(traj_file, mode='r')
    ase.io.write(save_to, t, 'xyz')
    

def numbers2indices(numbers, species_to_indices=[1,6,7,8]):
    return np.array([species_to_indices.index(n) for n in numbers.flatten()]).reshape(numbers.shape)

def numbers2atomicenergies(numbers, atomic_energies):
    return np.sum(np.array([atomic_energies[n] for n in numbers.flatten()]).reshape(numbers.shape), axis=1)