#!/bin/bash 
#SBATCH --account=def-durandau
#SBATCH --job-name=dynsyn_new
#SBATCH --cpus-per-task=6
#SBATCH --time=0-47:50
#SBATCH --array=16
#SBATCH --mem=128G
#SBATCH --mail-user=huiyi.wang@mail.mcgill.ca
#SBATCH --mail-type=ALL

export PYTHONPATH="$PYTHONPATH:/home/cheryl16/projects/def-durandau/cheryl16/DynSyn"

cd /home/cheryl16/projects/def-durandau/cheryl16/DynSyn

module purge                     # Clear any conflicting modules  
module load StdEnv/2023          # Load the standard environment  
module load scipy-stack/2024b    # Load the SciPy stack (includes numpy)  
module load gcc opencv/4.9.0 cuda/12.2 python/3.10 mpi4py mujoco/3.1.6

source ~/venvs/dynsyn_310/bin/activate

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline

wandb offline

python -m dynsyn.sb3_runner.runner -f configs/DynSyn/myofullwalk.json