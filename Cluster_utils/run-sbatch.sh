#!/usr/bin/env bash

#SBATCH -J HRL # name of the project
#SBATCH -p high # priority
#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH --mem 10gb
#SBATCH --workdir=/homedtic/lsteccanella/DeepRL/ # working directory project
#SBATCH -C intel #request intel node (those have infiniband) # intel node
#SBATCH -o /homedtic/lsteccanella/DeepRL/Cluster_utils/jobs/%N.%J.out # STDOUT # output to number of node number of job
#SBATCH -e /homedtic/lsteccanella/DeepRL/Cluster_utils/jobs/%N.%j.err # STDERR # output of the error

# set -x # output verbose
source /homedtic/lsteccanella/DeepRL/Cluster_utils/modules.sh
source /homedtic/lsteccanella/DeepRL/Cluster_utils/cluster_env_2/bin/activate
python -u /homedtic/lsteccanella/DeepRL/main.py "$@"

