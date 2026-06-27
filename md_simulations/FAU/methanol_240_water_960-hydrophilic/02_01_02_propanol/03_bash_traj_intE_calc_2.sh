#/bin/bash 

cwd="$(pwd)"  
echo " " >> traj_intE.txt
# echo $cwd


 for folder in intE*; do 
	cd ${folder}/01_intE*_aq/00-OUTPUT/
	# echo ${folder}/aq >> ../../../traj_intE.txt
	intE1_list=$(sed -n '2p' *converged.xyz)
	set -- $intE1_list
	intE1=$(echo $6 )
	echo ${folder}/aq $intE1 >> ../../../traj_intE.txt
	cd $cwd
	
	cd ${folder}/02_intE*_aq_zeo/00-OUTPUT/
	# echo ${folder}/aq_zeo >> ../../../traj_intE.txt
	intE2_list="$(grep 'ENERGY|' intE*.out)"
	set -- $intE2_list
	intE2=$(echo $9 )
	echo ${folder}/aq_zeo $intE2 >> ../../../traj_intE.txt
	cd $cwd
	
	cd ${folder}/03_intE*_g_sp/00-OUTPUT/
	# echo ${folder}/aq_zeo >> ../../../traj_intE.txt
	intE3_list="$(grep 'ENERGY|' intE*.out)"
	set -- $intE3_list
	intE3=$(echo $9 )
	echo ${folder}/g_sp $intE3 >> ../../../traj_intE.txt
	cd $cwd
	echo " " >> traj_intE.txt
done

### zeolite energy
ad_name="$(basename $PWD)"
cd /fs/ess/PAS2536/xiutinc/zeolite/FAU/intE_dft_4sioh/${ad_name}/00_intE_zeo/00-OUTPUT/
zeo_intE=$(sed -n '/ENERGY|/p' *.out|awk '{print $9}')
echo "00_intE_zeo" ${zeo_intE} >>$cwd/traj_intE.txt
cd $cwd

python /fs/ess/PAS2536/xiutinc/home/DFT/31_intE_dft_one_unit_osc/calc_intE_2.0_9.26.py >> traj_intE.txt