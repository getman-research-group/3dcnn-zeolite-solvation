#/bin/bash 

cwd="$(pwd)"  
echo " " >> traj_intE.txt
echo $cwd


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
	intE2_list="$(grep 'ENERGY|' *.out)"
	set -- $intE2_list
	intE2=$(echo $9 )
	echo ${folder}/aq_zeo $intE2 >> ../../../traj_intE.txt
	cd $cwd
	
	cd ${folder}/03_intE*_g_sp/00-OUTPUT/
	# echo ${folder}/aq_zeo >> ../../../traj_intE.txt
	intE3_list="$(grep 'ENERGY|' *.out)"
	set -- $intE3_list
	intE3=$(echo $9 )
	echo ${folder}/g_sp $intE3 >> ../../../traj_intE.txt
	cd $cwd
	echo " " >> traj_intE.txt
done

python calc_intE.py >> traj_intE.txt