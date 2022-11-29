#!/bin/bash
#SBATCH --job-name=inloc
#SBATCH --output=logs/inloc_algo_%j.log
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


set -e

. /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
module load MATLAB/2019b
module load SuiteSparse/5.1.2-foss-2018b-METIS-5.1.0

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

nvidia-smi

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo

CONFIG_NAME=${1:-main}
TMP_PARAMS=$(mktemp)

trap "rm -f ${TMP_PARAMS}" 0 2 3 15

# check if script is started via SLURM or bash
if [ -n "${SLURM_JOB_ID}" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT=$(realpath $(scontrol show job ${SLURM_JOBID} | awk -F= '/Command=/{print $2}' | cut -d' ' -f1))
    CURRENT_DIR="$( cd "$( dirname "${SCRIPT}" )" && pwd )"
else
    # otherwise: started with bash. Get the real location.
    CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Resolve libvl.so: cannot open shared object file: No such file or directory.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CURRENT_DIR/../../../functions/vlfeat/toolbox/mex/mexa64/

cat > ${TMP_PARAMS} <<- EOF
params_file = '$(realpath params.yaml)';
experiment_name = '${CONFIG_NAME}';

EOF

cd "${CURRENT_DIR}/../../../inLocCIIRC_demo"
code=$(cat ${TMP_PARAMS} inloc_demo_neural.m)
echo "${code}"
echo "${code}" | ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' matlab -nodesktop
