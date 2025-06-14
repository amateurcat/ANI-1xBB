import io, re, os, subprocess
import numpy as np
from matplotlib import pyplot as plt
import ase.io
from ase.io.xyz import write_xyz
#from rdkit import Chem
from .maxmin import find_interest
from .utils import in_tempdir

HARTREE_TO_KCALMOL = 627.509

MTD_N_SAMPLE = 10
MTD_KPUSH_PER_ATOM = 0.05
MTD_ALPHA = 1.0

XTB_MD_TEMP = 400.0   #K
XTB_MD_TOTAL_TIME = 1.0   #ps
XTB_MD_DUMP_INTERVAL = 10   #fs
XTB_MD_STEPLENGTH = 0.5   #fs

TMP = '/dev/shm'   #place to generate temp files
ETEMP = 1000.0


def write_opt_input_file(at1, at2):
    with open('opt_fix_%d-%d.inp'%(at1, at2),'w') as fw:
        fw.write(
'''
$fix
   atoms: %d,%d
$end
$opt
    engine=rf
    $end
'''%(at1, at2)
        )

def write_mdy_input_file(at1, at2, kpush, alpha=MTD_ALPHA, sample_steps=MTD_N_SAMPLE):
    with open('mdy_fix_%d-%d.inp'%(at1, at2),'w') as fw:
        fw.write(
'''
$constrain
   force constant=100
   distance: %d,%d, auto
$end
$metadyn
   save=%d
   kpush=%f
   alp=%f
$end
'''%(at1, at2, sample_steps, kpush, alpha)
        )

def write_md_input_file(at1, at2, temp=XTB_MD_TEMP, time=XTB_MD_TOTAL_TIME, dump=XTB_MD_DUMP_INTERVAL, steplength=XTB_MD_STEPLENGTH):
    # $fix did not work for MD, the only way to fix atom in MD is to use $constrain with a huge force constant
    # this will not affect the total energy
    with open('md_fix_%d-%d.inp'%(at1, at2),'w') as fw:
        fw.write(
'''
$constrain
   force constant=0.1
   distance: %d,%d, auto
$end
$md
  temp=%.2f
  time= %.1f
  dump= %.1f
  step=%.2f
  hmass=1.0
  shake=0
$end
$wall
  potential=logfermi
  sphere: auto, all
$end
'''%(at1, at2, temp, time, dump, steplength)
        )

def run_xtb_opt(mol, at1, at2):
    with in_tempdir(basedir=TMP):
        ase.io.write('temp.xyz', mol)
        write_opt_input_file(at1, at2)
        #os.system('xtb --opt --input opt_fix_%d-%d.inp temp.xyz'%(at1, at2))
        #print(os.getcwd())
        #print(os.listdir())
        #print('export OMP_NUM_THREADS=1&&xtb --opt --etemp %f -P 1 --input opt_fix_%d-%d.inp temp.xyz'%(at1, at2, ETEMP))
        subprocess.run('export OMP_NUM_THREADS=1&&xtb --opt --etemp %f -P 1 --input opt_fix_%d-%d.inp temp.xyz'%(ETEMP, at1, at2), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print(os.listdir())
        try:
            m = ase.io.read('xtbopt.xyz')
            with open('xtbopt.xyz','r') as f:
                energy = float(re.search('-?[0-9]+\.?[0-9]+',f.readlines()[1])[0])
        except FileNotFoundError:
            print('file not found')
            m = energy = 'Failed'
        
    #print('opt finished')
    
    return m, energy

def run_xtb_mdy(mol, at1, at2):
    with in_tempdir(basedir=TMP):
        ase.io.write('temp.xyz', mol)
        write_mdy_input_file(at1, at2, kpush=len(mol)*MTD_KPUSH_PER_ATOM, alpha=MTD_ALPHA, nsteps=MTD_N_SAMPLE)
        #os.system('xtb --md --input md_fix_%d-%d.inp temp.xyz'%(at1, at2))
        subprocess.run('xtb --md --input md_fix_%d-%d.inp temp.xyz --etemp %f'%(at1, at2, ETEMP), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.rename('xtb.trj','xtb.xyz')
        ms = ase.io.read('xtb.xyz',':')
        with open('xtb.xyz','r') as f:
            energies = []
            for l in f.readlines():
                t = re.search(' energy: -?[0-9]+\.?[0-9]+',l)
                if t:
                    energies.append(float(re.search('-?[0-9]+\.?[0-9]+',t[0])[0]))
    #print('meta dynamic finished')
    return ms, energies

def run_xtb_md(mol, at1, at2):
    with in_tempdir(basedir=TMP):
        ase.io.write('temp.xyz', mol)
        write_md_input_file(at1, at2, temp=XTB_MD_TEMP, time=XTB_MD_TOTAL_TIME, dump=XTB_MD_DUMP_INTERVAL, steplength=XTB_MD_STEPLENGTH)
        os.environ['OMP_NUM_THREADS'] = '1'
        #os.system('xtb --md --input md_fix_%d-%d.inp temp.xyz'%(at1, at2))
        subprocess.run('export OMP_NUM_THREADS=1&&xtb --CMA --md --etemp %f -P 1 --input md_fix_%d-%d.inp temp.xyz'%(ETEMP, at1, at2), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        '''
        os.system('cat xtb.trj')
        print("************************************")   
        os.system('ls')
        with open('xtb.trj','r') as f:   #MD cannot run for some molecules!
            print(f.read())
        with open('md_fix_%d-%d.inp'%(at1, at2),'r') as f: 
            print(f.read())
        with open('temp.xyz','r') as f: 
            print(f.read())
        '''

        os.rename('xtb.trj','xtb.xyz')   
        ms = ase.io.read('xtb.xyz',':')
        with open('xtb.xyz','r') as f:
            energies = []
            for l in f.readlines():
                t = re.search(' energy: -?[0-9]+\.?[0-9]+',l)
                if t:
                    energies.append(float(re.search('-?[0-9]+\.?[0-9]+',t[0])[0]))

    #print('md finished')
    return ms, energies


def stepwise_opt_a_bond(mol, mol_name, at1, at2, ratio=1.1, nstep=2, run_mdy=False, run_md=False, plot_energy=False, do_maxmin=False, prefix=''):
    
    d0 = mol.get_distance(int(at1)-1, int(at2)-1)
    opt_traj = io.StringIO()   #only conformers got from opt step
    total_traj = io.StringIO()   # all conformers generated
    

    energy_plot = []   #collection of energies got from OPTIMIZE steps, energies from MD steps will NOT be poltted!
    all_energies = []   #collection of energies got from both optimize steps AND MD steps


    for d in np.linspace(d0, d0*ratio, nstep):
        mol.set_distance(at1-1, at2-1, d)
        mol, energy = run_xtb_opt(mol, at1, at2)
        if mol == 'Failed':
            break
        energy_plot.append(energy)
        all_energies.append(energy)
        write_xyz(total_traj, mol)
        write_xyz(opt_traj, mol)

        if run_md:
            ms, energies = run_xtb_md(mol, at1, at2)
            all_energies += energies
            write_xyz(total_traj, ms)
        elif run_mdy:
            ms, energies = run_xtb_mdy(mol, at1, at2)
            all_energies += energies
            write_xyz(total_traj, ms)
        try:
            mol = ms[-1]
        except NameError:
            pass
        #print('one step finished')

    if mol == 'Failed':
        print("%s Failed!"%(mol_name))
    else:
        #trajectory file contain only conformers geneated from opt step
        opt_traj.seek(0)
        with open(os.path.join(prefix,"%s_%d-%d_opt.xyz"%(mol_name, at1,at2)), 'w') as f:
            f.write(opt_traj.read())

        #trajectory file contain all conformers generated 
        total_traj.seek(0)
        with open(os.path.join(prefix,"%s_%d-%d_total.xyz"%(mol_name, at1,at2)), 'w') as f:
            f.write(total_traj.read())

        #save all energies to npy file
        np.save(os.path.join(prefix,"%s_%d-%d_energies.npy"%(mol_name, at1,at2)),np.array(all_energies)*HARTREE_TO_KCALMOL)

        #this is for debug
        '''
        d_curve = []
        for c in ase.io.read(os.path.join(prefix,"%s_%d-%d_total.xyz"%(mol_name, at1,at2)),':'):
            d_curve.append( c.get_distance(at1-1, at2-1) )
        plt.plot(list(range(len(d_curve))), d_curve)
        plt.savefig(os.path.join(prefix,'d_curve.png'))
        plt.cla()
        '''

        if plot_energy:
            plt.plot(np.linspace(d0, d0*ratio, nstep), (np.array(energy_plot)-energy_plot[0])*HARTREE_TO_KCALMOL)   #plot the energy change w.r.t. the original conformer, only energy on opt steps!
            plt.xlabel('distance between atom %d and %d'%(at1,at2))
            plt.ylabel('energy change(Kcal/mol)')
            plt.savefig(os.path.join(prefix,"%s_%d-%d.png"%(mol_name, at1,at2)))
            plt.cla()
        
        if do_maxmin:
            find_interest(os.path.join(prefix,"%s_%d-%d_total.xyz"%(mol_name, at1,at2)))
        
        print("%s finished!"%(mol_name))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    #parser.add_argument('--smile_input', type=str)
    parser.add_argument('--xyz_input', type=str)
    parser.add_argument('--scan_all', action='store_true', default=False)
    parser.add_argument('--at1', type=int)
    parser.add_argument('--at2', type=int)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--nstep', type=int)
    parser.add_argument('--mdy', action='store_true', default=False)
    parser.add_argument('--md', action='store_true', default=False)
    parser.add_argument('--plot_energy', action='store_true', default=False)
    parser.add_argument('--do_maxmin', action='store_true', default=False)
    args = parser.parse_args()

    mol = ase.io.read(args.xyz_input)
    mol_name = args.xyz_input.split("/")[-1].split(".")[0]

    stepwise_opt_a_bond(mol, mol_name, args.at1, args.at2, ratio=args.ratio, nstep=args.nstep, run_mdy=args.mdy, run_md=args.md, plot_energy=False, do_maxmin=args.do_maxmin)
       
