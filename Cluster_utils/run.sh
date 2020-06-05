E_PATH='/homedtic/lsteccanella/DeepRL/Protocols/TEST/SIL_POSITION_TRANSFER_3/'
prefix='/homedtic/lsteccanella/DeepRL/Protocols/'
suffix=

for i in $(find $E_PATH -name '*.py' -type f);
do 
	for j in 2
	do	
		SEED=$j
		PROT_PATH=${i#"$prefix"}
		PROT_PATH=${PROT_PATH%.*}
		PROT_PATH="$(tr / . <<<$PROT_PATH)"
		PROT_PATH_SEED="$PROT_PATH $SEED"
		echo $PROT_PATH_SEED
		sbatch --exclude=node[022-030] -C intel /homedtic/lsteccanella/DeepRL/Cluster_utils/run-sbatch.sh $PROT_PATH_SEED
	done
	
done

