#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 3

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=1:00:00

module load python3/3.8.10
module load pytorch/1.13.1
# # conda info -e
conda activate py3.8
export LD_LIBRARY_PATH=/share/pkg.7/miniconda/4.9.2/install/lib/
conda info -e
which conda
which python
echo $PATH
# conda info -e
export LD_LIBRARY_PATH=/share/pkg.7/miniconda/4.9.2/install/lib/
conda info -e
which conda
which python
echo $PATH

dataset='MRI'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'XAIalign') 

test_envs="1 2 3"
gpu_ids="0"
# i=1  # ERM
i=$1  # VREx
FOLD=$2
cohort="NACC_NC_MCI_AD"
# cohort="NACC_NC_MCI_AD"
data_dir="/projectnb/ivc-ml/dlteif/NACC_NC_MCI_AD_folds/fold_${FOLD}/train.csv"
# data_dir="/projectnb/ivc-ml/dlteif/ayan_datasets/neuropath/FHS_NC_MCI_AD_np.csv"
max_epoch=100
# max_epoch=70
steps_per_epoch=60
batch_size=1
# batch_size=4
net='unet3d'
task='mri_dg'
# output='output_dir/attention_maps'
output='output_dir/shap_maps'
checkpoint_freq=1
classifier='gap'
lambda=1e-05
path="output_dir/${algorithm[i]}/${net}_${classifier}_attention_${cohort}aug_minmax_normalize_lr0.0008_${FOLD}"
# path="output_dir/${algorithm[i]}/${net}_${classifier}_pretrained_attention_${cohort}aug_minmax_normalize_lr0.0008layer_feats_classifier_l2_start0_lambda${lambda}_gamma0.0_${FOLD}"
resume="${path}/model_best.pkl"
# resume="/data_1/diala/DeepDG_checkpoints/output_dir/9x11x9/${algorithm[i]}/${net}_${classifier}_attention_NACC_ADNI_NC_MCI_AD_minmax_normalize_lr0.0008layer_feats_avgpool_l2_start0_lambda0.01_gamma0.0_${FOLD}/model_best.pkl"
# resume="output_dir/${algorithm[i]}/${cohort}_${FOLD}/model_best.pkl"


# MLDG 
python -m memory_profiler visualize_shap.py --data_dir $data_dir --batch_size $batch_size --net $net --classifier $classifier --attention --task $task --output $output --num_classes 3 --blocks 4 --attention \
--fold ${FOLD} --dataset $dataset --cohort $cohort --algorithm ${algorithm[i]} --steps_per_epoch ${steps_per_epoch} --test_envs ${test_envs} \
--N_WORKERS 4 --save_model_every_checkpoint --checkpoint_freq ${checkpoint_freq} --lr 8e-4 --N_WORKERS 4 --gpu_id ${gpu_ids} \
--align_start 0 --align_lambda 0.2 --align_gamma 0 --layer classifier --num_classes 3 --resume=${resume} --eval --write_raw_score #--config=${path}/config.json # # --eval  #--mldg_beta 10

