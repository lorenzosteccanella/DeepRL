#!/usr/bin/env bash

source /homedtic/lsteccanella/DeepRL/Cluster_utils/modules.sh
source /homedtic/lsteccanella/DeepRL/Cluster_utils/cluster_env/bin/activate
python -u /homedtic/lsteccanella/DeepRL/main.py "$@"

