# Sourced by every job: environment, paths, headless rendering.
module load miniconda >/dev/null 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nfs/roberts/project/pi_dr877/ak3288/envs/dem
export PROJ=/nfs/roberts/project/pi_dr877/ak3288 SCR=/nfs/roberts/scratch/pi_dr877/ak3288
export ROOT=$PROJ/apollo-lab REPO=$PROJ/apollo-lab/rua-harness
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_CACHE=$SCR/cache/hf PYTHONPATH=$PROJ/apollo-lab/rua-harness
export RUA_DATA=$SCR/datasets RUA_FRAMES=$SCR/frames
export ALL_TASKS="CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet"
cd $PROJ/apollo-lab/runs
