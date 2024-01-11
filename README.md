# InLoc

This repository unifies codes needed for InLoc pipeline's expected
dataset format transformation, the pipeline itself, and
contains also codes for experiments from https://github.com/Auratons/master_thesis.
The repository is linked and called (for InLoc pipeline dataset format generation)
from https://github.com/Auratons/neural_rendering.

The repository contains git submodules, so either clone the repository
with `--recurse-submodules` option or inside of the folder run
`git submodule init && git subbmodule update --recursive`.

## Repository structure

The repository contains git submodules which are either targeting original
implementations or forks of those, namely `inLocCIIRC_demo`, `inLocCIIRC_dataset`,
and `functions/inLocCIIRC_utils`. The `dvc` folder is the main entrypoint to
experiments. (The folder name is a remnant of a trial to use [DVC](https://dvc.org)
which proven itself to be an unsuitable tool for datasets comprised of many small
files.) The `datasets` folder should contain the data and their transformations.
The `notebooks` folder contain a Jupyter notebook with experimental code samples
and explorations.

## Dependencies & Runtime

The runtime for the project was Slurm-based compute cluster with graphical capabilities
operated by [Czech Institute of Informatics, Robotics and Cybernetics](https://cluster.ciirc.cvut.cz).
Thus, in `dvc/scripts` folder, there are mentions of SBATCH directives meant as Slurm
scheduler limits and compute requirements for various workloads.

The pipeline runs with Matlab, the version 2019b was used. Python codes used can be
ran with the Python virtual environment defined in https://github.com/Auratons/neural_rendering.

In the scripts also binaries `time` as `gnu-time` are mentioned.

## Data

The data are described in https://github.com/Auratons/neural_rendering, for
the dataset format generation sources generated from the neural rendering repository are used an stored in its `datasets` folder mainly, here there are stored InLoc pipeline's
intermediate representations, verifications step renders and evaluations, as specified
in the dvc folder.

## Running codes

Prepare conda environment from `environment.yml` in the neural rendering repository,
go to specific subfolder of `dvc/pipeline-*` depending on which dataset should be
targeted and run `sbatch ../scripts/<SCRIPT_NAME> <CONFIG_NAME>`.
Scripts always read `params.yaml` file and pick proper configuration key `<CONFIG_NAME>`,
typically it defaults to `main` as there is only one configuration in the file.
