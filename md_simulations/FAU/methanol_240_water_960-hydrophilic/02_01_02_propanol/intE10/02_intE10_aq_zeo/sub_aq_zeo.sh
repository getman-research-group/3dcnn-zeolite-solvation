#!/bin/bash
#SBATCH --job-name=zl02_01_02_propanol
#SBATCH --ntasks=24
##SBATCH --ntasks-per-node=12
##SBATCH --output=rfm_cp2k_test_mpi_job.out
#SBATCH --error=rfm_cp2k_test_mpi_job.err
#SBATCH --time=8:10:0
##SBATCH -p parallel
#SBATCH --account PAS2536

module load gcc/12.3.0
module load openmpi/5.0.2
module load cp2k/2023.2

# Check module environment
module list
echo MODULEPATH=$MODULEPATH 1>&2



# Check slurm environment
echo SLURM_NTASKS=$SLURM_NTASKS
echo SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE
echo SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
echo SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
echo SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
echo SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE
echo

export SCR=/fs/scratch/PAS2536
#define the run directory on the scratch disk:
export WRK=$SCR/xiutinc/$SLURM_JOB_ID.$SLURM_JOB_NAME

mkdir -p $WRK
echo $WRK
SCRATCH_DIR=$WRK
cwd="$(pwd)"




# file needed for valence density simulation



# rename xyz file and modify inp file
cd ${cwd}
filename=$(basename *.xyz .xyz)
atom_number=$(head -n 1 *.xyz)
mv *.inp ${filename}.inp
sed -i "s/ADSORBATE_NAME/${filename}/" *.inp
sed -i "s/ADSORBATE_NUMBER/${atom_number}/" *.inp
cd ${cwd}
sed -i "s/intE10_aq_zeo/${filename}/" sub_aq_zeo.sh	


# Starting the CP2K calculation 
# cd ${cwd}  
# module purge

# export MODULEPATH=/software/ModuleFiles/modules/linux-centos8-ivybridge:$MODULEPATH
# module load cp2k/9.1-gcc/9.5.0-mpi-openmp 

# # export MODULEPATH=/software/ModuleFiles/modules/linux-centos8-ivybridge:$MODULEPATH
# # module load cp2k/7.1-gcc/8.3.1-mpi

module list 
# export OMP_NUM_THREADS=1

# # Creating the CP2K scratch directory
# SCRATCH_DIR="/scratch1/$USER/CP2K-${PBS_JOBID}/"

SCRATCH_DIR=$WRK

mkdir -p ${SCRATCH_DIR}
cp ./* ${SCRATCH_DIR}/.
echo ${SCRATCH_DIR} > ./zb-TO-SCRATCH.dir
echo ./ > ${SCRATCH_DIR}/zb-TO-PBS-WORKDIR.dir
echo '' > ./zc-${PBS_JOBID%.*}.JOBID
 
# Running CP2K
cd ${SCRATCH_DIR}
NAME=$(basename *.inp .inp)
# mpirun -n 20 cp2k.popt -i ${NAME}.inp -o ${NAME}.out
srun cp2k.popt  -i ${NAME}.inp -o ${NAME}.out
# Copying over all the raw output files from CP2K
cd ${cwd}
mkdir -p ./00-OUTPUT 
cp ${SCRATCH_DIR}/* ./00-OUTPUT
if [[ -e ./00-OUTPUT/zb-TO-PBS-WORKDIR.dir ]]; then 
	rm ./00-OUTPUT/zb-TO-PBS-WORKDIR.dir
fi  

# Creating converged structure .xyz file 
# cd ${cwd}/00-OUTPUT
# file_xyz=$(find . -type f -name "*pos-1.xyz")
# number_atoms=`expr $(head -1 ${file_xyz}) + 2`
# tail -${number_atoms} ${file_xyz} >> ${file_xyz%.*}_converged.xyz
# echo ${file_xyz%.*}

ENDING VALENCE DENSITY CALCULATION
echo ''
echo ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # '
echo ''
echo ' ENDING THE VALENCE DENSITY CALCULATION!'
echo ''
echo ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # '
echo ''

qstat -xf $PBS_JOBID
rm -f core.*