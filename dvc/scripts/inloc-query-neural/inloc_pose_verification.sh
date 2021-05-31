#!/bin/bash
#SBATCH --job-name=inloc_pose_verification
#SBATCH --output=logs/inloc_pose_verification_%j.log
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load MATLAB/2018a
module load SuiteSparse/5.1.2-foss-2018b-METIS-5.1.0

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

CONFIG_NAME=$1
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
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$CURRENT_DIR/../../../inLocCIIRC_demo/functions/vlfeat/toolbox/mex/mexa64/

cat > ${TMP_PARAMS} <<- EOF

	params = struct();

	params.input.dir              = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.input.dir')';
	params.input.topK_matfile     = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.input.topK_matfile')';
	params.input.imgformat        = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.input.imgformat')';

	params.output.dir             = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.output.dir')';
	params.output.txtname         = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.output.txtname')';
	params.output.matname         = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.output.matname')';
	params.output.synth.dir       = $(cat params.yaml | yq -r '.'${CONFIG_NAME}'.output.synth_dir');
	params.output.synth.matformat = '$(cat params.yaml | yq -r '.'${CONFIG_NAME}'.output.synth_matformat')';

EOF

cd $CURRENT_DIR/../../../inLocCIIRC_demo
cat startup.m ${TMP_PARAMS} inloc_neural_pose_verification.m | ~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' matlab -nodesktop
