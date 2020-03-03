E_PATH='/homedtic/lsteccanella/DeepRL/Protocols/Exploration_Effort/repr_states/Key_Door/'
prefix='/homedtic/lsteccanella/DeepRL/Protocols/'
suffix=
for i in $(find $E_PATH -name '*.py' -type f);
do 
	PROT_PATH=${i#"$prefix"}
	PROT_PATH=${PROT_PATH%.*}
	PROT_PATH="$(tr / . <<<$PROT_PATH)"
	sbatch /homedtic/lsteccanella/DeepRL/Cluster_utils/run-sbatch.sh $PROT_PATH
	
done


