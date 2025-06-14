import os
import numpy as np


total_jobs = 90920
def name_finder(dataset):

    names = []
    with open(dataset,'r') as fr:
        l = fr.read().split("$$$$\n")   #this \n is necessary
        del l[-1]
        for s in l:
            context = s.split('\n')
            m = int(context[0])
            at1,at2 = context[-3].split(' ')
            names.append('m%d_%d_%d'%(m,int(at1),int(at2)))
    
    return np.array(names)
    
        
def jobfile_generator(total=total_jobs, target_dir='/home/shuhaozh/reaction_database/bond_breaking3.1/jobs/for_r2'):

    for i in np.arange(0,total,1500):
        with open(os.path.join(target_dir,'r1_%d-%d.job'%(i,i+1500)),'w') as f:
            f.write(
'''#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 9:00:00

#echo commands to stdout
set -x

# move to working directory
# this job assumes:
# - all input data is stored in this directory 
# - all output should be stored in this directory

export OMP_NUM_THREADS=1

cd /home/shuhaozh/reaction_database/bond_breaking3.1
conda activate ani_network
python multirun.py --start %d --stop %d
'''%(i,i+1500))

jobfile_generator()