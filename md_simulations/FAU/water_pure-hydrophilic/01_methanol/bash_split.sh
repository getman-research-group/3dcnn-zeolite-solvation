#/bin/bash 

cutoff=7
adsorbate_file=01_methanol

cp ../../fep_generate/${adsorbate_file}/01_geo_opt/00-OUTPUT/*converged.xyz ./
cp ../../fep_generate/${adsorbate_file}/21_npt_nvt/{dump_nvt_samp.lammpsdata.lammpstrj,data_nvt_samp.lammpsdata} ./
cp /zfs/curium/xiutinc/zeolite/FAU/intE_dft_one_unitcell/03_01_1_3_propanediol/{bash_g_sp.sh,bash_traj_intE_calc.sh,calc_intE.py,traj_2_intE.py,zeolite_ad_g.py} ./

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


# for each traj make xyz_zeoad_aq and xyz_zeo_aq
for traj_file in *.traj; do
	cd ${cwd}
	echo $traj_file
	foldername=$(basename ${traj_file} .traj)
	mkdir -p ${foldername}/{01_${foldername}_aq,02_${foldername}_aq_zeo}
	python traj_2_intE.py $traj_file converged_one_unit.xyz $cutoff
	
	mv $traj_file ${foldername}/
	mv ${foldername}_aq.xyz ${foldername}/01_${foldername}_aq
	mv ${foldername}_aq_zeo.xyz ${foldername}/02_${foldername}_aq_zeo
	
	# adsorbate relax in aqueous phase
	cd ${foldername}/01_${foldername}_aq
	cp /home/xiutinc/DFT/31_intE_dft_one_unit/01_intE_aq/* ./
	sed -i "s/GAS_XYZ/${gas_xyz}/" *.inp
	sed -i "s/aq_ad_geo/${job_name_aq_ad_geo}/" sub_aq.sh
	qsub sub_aq.sh 
	
	# single point of zeolite reference in aqueous phase
	cd ../02_${foldername}_aq_zeo
	cp /home/xiutinc/DFT/31_intE_dft_one_unit/02_intE_aq_zeo/* ./
	sed -i "s/aq_zeo_sp/${job_name_aq_zeo}/" sub_aq_zeo.sh
	qsub sub_aq_zeo.sh
	
done >> ./traj_intE.txt
	