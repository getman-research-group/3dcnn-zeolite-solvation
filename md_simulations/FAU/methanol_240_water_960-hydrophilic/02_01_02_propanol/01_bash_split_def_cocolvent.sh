#/bin/bash 

cutoff=7
adsorbate_file=$(basename $(pwd))

###### get gas phase file from DFT calculation, get lammps data file and trajectory file from MD simulation###
## 1. *.converged.xyz 
## 2. dump_nvt_samp.lammpsdata.lammpstrj
## 3. data_nvt_samp.lammpsdata
cp /fs/ess/PAS2536/xiutinc/zeolite/FAU/fep_generate/${adsorbate_file}/01_geo_opt/00-OUTPUT/*converged.xyz ./
# cp /fs/ess/PAS2536/xiutinc/zeolite/FAU/fep_generate/${adsorbate_file}/21_npt_nvt/{dump_nvt_samp.lammpsdata.lammpstrj,data_nvt_samp.lammpsdata} ./
# cp /fs/ess/PAS2536/xiutinc/zeolite/FAU_cosolvent/03_water_methanol_240_960/02_lammps_generate/10_free/${adsFree}/21_npt_nvt/data_nvt_samp.lammpsdata ./
# cp /fs/ess/PAS2536/xiutinc/zeolite/FAU_cosolvent/03_water_methanol_240_960/02_lammps_generate/10_free/${adsFree}/32_inte/*.i*/dump_nvt_samp.lammpsdata.lammpstrj ./
##### get running file and python script to generate xyz file to run intE
## 1. 2_bash_g_sp_cosolvent.sh,03_bash_traj_intE_calc_2.sh
## 2. calc_intE_2.0_9.26.py,traj_2_intE_cosol_3.0.py,zeolite_ad_g.py
#### detial is 1. use 01_bash_split file to calculate electronic energy of zeo+ad+sol and zeo+sol 
####           2. use 02_bash_g_sp_cosolvent.sh to calculate electronic energy of zeo+ad 
####           3. use 03_bash_traj_intE_calc_2.sh to write down all energy values and calculate the average value.
cp /fs/ess/PAS2536/xiutinc/home/DFT/31_intE_dft_one_unit_osc/{02_bash_g_sp_cosolvent.sh,03_bash_traj_intE_calc_2.sh,calc_intE_2.0_9.26.py,traj_2_intE_cosol_3.0.py,zeolite_ad_g.py} ./

cwd="$(pwd)" 

job_name_aq_ad_geo="adl$(basename $PWD)"
job_name_aq_zeo="zl$(basename $PWD)"

python zeolite_ad_g.py *converged.xyz converged_one_unit.xyz


cwd="$(pwd)" 
# split trajectory to 10 xyz file
sed -i 's/\r$//' *.lammpstrj
number_lines=`expr $(sed -n 4p  *.lammpstrj) + 9`
split -l ${number_lines} -d --additional-suffix=.traj *.lammpstrj intE
rm *00.traj

sed -i 's/\r$//' *converged.xyz
gas_xyz=`expr $(sed -n 1p  converged_one_unit.xyz) + 1`


# for each traj generate xyz_zeoad_aq and xyz_zeo_aq
for traj_file in *.traj; do
	cd ${cwd}
	echo $traj_file
	foldername=$(basename ${traj_file} .traj)
	mkdir -p ${foldername}/{01_${foldername}_aq,02_${foldername}_aq_zeo}
	python traj_2_intE_cosol_3.0.py $traj_file converged_one_unit.xyz $cutoff 
	
	mv $traj_file ${foldername}/
	mv ${foldername}_aq.xyz ${foldername}/01_${foldername}_aq
	mv ${foldername}_aq_zeo.xyz ${foldername}/02_${foldername}_aq_zeo
	
	# adsorbate relax in aqueous phase
	cd ${foldername}/01_${foldername}_aq
	cp /fs/ess/PAS2536/xiutinc/home/DFT/31_intE_dft_one_unit_osc/01_intE_aq/* ./ ###??? folder need to edit
	sed -i "s/GAS_XYZ/${gas_xyz}/" *.inp
	sed -i "s/aq_ad_geo/${job_name_aq_ad_geo}/" sub_aq.sh
	sbatch sub_aq.sh 
	
	# single point of zeolite reference in aqueous phase
	cd ../02_${foldername}_aq_zeo
	cp /fs/ess/PAS2536/xiutinc/home/DFT/31_intE_dft_one_unit_osc/02_intE_aq_zeo/* ./ ###??? folder need to edit
	sed -i "s/aq_zeo_sp/${job_name_aq_zeo}/" sub_aq_zeo.sh
	sbatch sub_aq_zeo.sh
	
done >> ./traj_intE.txt
	