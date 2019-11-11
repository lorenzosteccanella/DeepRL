for var in "$@"
do
    sbatch /homedtic/lsteccanella/DeepRL/Cluster_utils/run-sbatch.sh "$var"
    #sleep 5
done
