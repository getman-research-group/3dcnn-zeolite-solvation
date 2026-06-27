#/bin/bash 


# job_name
job_name="adg$(basename $PWD)"

cwd="$(pwd)"
file_line=`expr $(sed -n 1p  converged_one_unit.xyz) + 2`
atom_number=$(head -n 1 converged_one_unit.xyz)

# folder=intE01
for folder in intE*; do
	cd ${folder}
	rm -r 03_${folder}_g_sp/
	mkdir -p 03_${folder}_g_sp/
	cd 03_${folder}_g_sp/
	cp ../01*/00*/*converged.xyz ./
	
	echo $atom_number > ${folder}_g_sp.xyz
	echo '' >> ${folder}_g_sp.xyz
	sed -n "3,${file_line}"p *converged.xyz >> ${folder}_g_sp.xyz
	
	rm *converged.xyz
	cp -r /fs/ess/PAS2536/xiutinc/home/DFT/31_intE_dft_one_unit_osc/03_intE_g_sp/* ./
	ls
	sed -i "s/ad_g_sp/${job_name}/" sub_ad_g_sp.sh
	sbatch sub_ad_g_sp.sh
	cd $cwd
	
done