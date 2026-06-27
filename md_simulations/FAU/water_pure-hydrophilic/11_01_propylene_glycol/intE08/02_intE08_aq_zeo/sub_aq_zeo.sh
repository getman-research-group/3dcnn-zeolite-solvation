#/bin/bash 
#PBS -N zl11_01_propylene_glycol
#PBS -l select=1:ncpus=20:mpiprocs=20:mem=120gb:interconnect=fdr,walltime=72:00:00
#PBS -q work1 
#PBS -j oe
#PBS -m n
#PBS -M xiutinc@g.clemson.edu

echo ''
echo ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # '
echo ''
echo ' STARTING THE CALCULATION!'
echo ''
echo ' # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # '
echo '' 


qstat -xf $PBS_JOBID 

cd $PBS_O_WORKDIR

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
sed -i "s/PROJECT_NAME/${filename}/" sub_geo.sh	


# Starting the CP2K calculation 
cd ${cwd}
module purge

export MODULEPATH=/software/ModuleFiles/modules/linux-centos8-ivybridge:$MODULEPATH
module load cp2k/9.1-gcc/9.5.0-mpi-openmp 

# export MODULEPATH=/software/ModuleFiles/modules/linux-centos8-ivybridge:$MODULEPATH
# module load cp2k/7.1-gcc/8.3.1-mpi

module list 
export OMP_NUM_THREADS=1

# Creating the CP2K scratch directory
SCRATCH_DIR="/scratch1/$USER/CP2K-${PBS_JOBID}/"
mkdir -p ${SCRATCH_DIR}
cp ./* ${SCRATCH_DIR}/.
echo ${SCRATCH_DIR} > ./zb-TO-SCRATCH.dir
echo ./ > ${SCRATCH_DIR}/zb-TO-PBS-WORKDIR.dir
echo '' > ./zc-${PBS_JOBID%.*}.JOBID
 
# Running CP2K
cd ${SCRATCH_DIR}
NAME=$(basename *.inp .inp)
mpirun -n 20 cp2k.popt -i ${NAME}.inp -o ${NAME}.out

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