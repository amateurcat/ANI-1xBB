'''
Here implements our final decision for bond breaking process
Instead of elongate the bond evenly, we first elnogate the bond 
to 2x origin bondlength within 10 steps, and then elongate from 2x bondlength
to 3x bondlength within 5 steps, therefore more attention would be paid to 1x to 2x
range. Also, instead of filter all comformers together after the whole process, 
here we filter after each MD period, and select 10 conformers from each period
so totally for each bond we need 15 opt steps and 15 MD steps,
each opt step generate 1 snapshot, each MD period generate 10 snapshots
totally got 165 snapshots from each bond breaking process
This script is only for demo or test run, it accept .xyz input
'''
from BondBreaking.breaker import *
from BondBreaking.maxmin import *


def stepwise_opt_a_bond(mol, mol_name, at1, at2, do_MD=True, plot_energy=False, prefix=''):
    
    d0 = mol.get_distance(int(at1)-1, int(at2)-1)
    total_traj = io.StringIO()   # all conformers generated
    

    energy_plot = []   #collection of energies got from OPTIMIZE steps

    distances = np.linspace(d0, 2 * d0, 10, endpoint=False).tolist() + \
               np.linspace(2 * d0, 3 * d0, 5, endpoint=False).tolist()
    print('start')
    for d in distances:
        print(d)
        #elongate
        mol.set_distance(at1-1, at2-1, d)
        #run opt
        mol, energy = run_xtb_opt(mol, at1, at2)
        print(energy)
        if mol == 'Failed':
            break
        energy_plot.append(energy)
        write_xyz(total_traj, [mol])

        #run MD
        if do_MD:
            ms, energies = run_xtb_md(mol, at1, at2)
            dmat = bpdist(getcoords(ms))
            idx = sorted([i for i, _ in maxmin(dmat, seed=1, mindist=0.05)][:10])
            filtered = []
            for i in idx:
                filtered.append(ms[i])
            write_xyz(total_traj, filtered)

        try:
            mol = ms[-1]
        except NameError:
            pass
        #print('one step finished')

    if mol == 'Failed':
        print("%s Failed!"%(mol_name))
    else:
        #trajectory file contain all conformers generated 
        total_traj.seek(0)
        with open(os.path.join(prefix,"%s_%d-%d_total.xyz"%(mol_name, at1,at2)), 'w') as f:
            f.write(total_traj.read())

        if plot_energy:
            plt.plot(distances, (np.array(energy_plot)-energy_plot[0])*HARTREE_TO_KCALMOL)   #plot the energy change w.r.t. the original conformer, only energy on opt steps!
            plt.xlabel('distance between atom %d and %d'%(at1,at2))
            plt.ylabel('energy change(Kcal/mol)')
            plt.savefig(os.path.join(prefix,"%s_%d-%d.png"%(mol_name, at1,at2)))
            plt.cla()
        
        print("%s finished!"%(mol_name))

if __name__ == '__main__':
    ######################################################################
    #The number for at1 and at2 is the number of line in coordinate parts#
    #of the input xyz file! i.e. i-th atom show in the xyz file!         #
    #use visualization tools to find atoms you want to focus first!      #
    ######################################################################
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--xyz', type=str)
    parser.add_argument('--at1', type=int)
    parser.add_argument('--at2', type=int)
    parser.add_argument('--do_MD', action='store_true', default=False)
    parser.add_argument('--plot_energy', action='store_true', default=False)
    args = parser.parse_args()

    mol = ase.io.read(args.xyz)

    mol_name = args.xyz.split("/")[-1].split(".")[0]

    stepwise_opt_a_bond(mol, mol_name, args.at1, args.at2, do_MD=args.do_MD, plot_energy=args.plot_energy)
