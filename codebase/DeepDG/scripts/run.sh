#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 2

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

dataset='MRI'
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx' 'XAIalign' 'Vanilla' 'GroupDRO' 'OrgMixup') 

test_envs="1 2 3"
# test_envs="3"
gpu_ids="0"
# i=1  # ERM
i=$1  # VREx
FOLD=$2
# cohort="NACC_ADNI_OASIS"
cohort="NACC_NC_MCI_AD"
data_dir="/projectnb/ivc-ml/dlteif/${cohort}_folds/fold_${FOLD}"
max_epoch=60
# max_epoch=70
accum_iter=8
steps_per_epoch=200
batch_size=2
# batch_size=4
net='unet3d'
task='mri_dg'
# output='/data_1/dlteif/echo_output_dir'
output='output_dir'
checkpoint_freq=1
classifier='gap'
loss='l2'
lambda=5e-05
# resume="output_dir/${algorithm[i]}/${net}_${classifier}_${cohort}_${FOLD}start0_lambda0.2_gamma0.0/model_best.pkl"
# path="output_dir/9x11x9/${algorithm[i]}/${net}_${classifier}_attention_${cohort}aug_minmax_normalize_lr0.0008layer_feats_classifier_l2_start0_lambda0.01_gamma0.0_${FOLD}"
# resume="/data_1/diala/DeepDG_checkpoints/${output}/${algorithm[i]}/${net}_${classifier}_attention_${cohort}_minmax_normalize_lr0.0008layer_feats_classifier_l2_start0_lambda${lambda}_gamma0.0_${FOLD}/model_best.pkl"
path="${output}/${algorithm[i]}/${net}_${classifier}_pretrained_attention_${cohort}aug_minmax_normalize_lr0.0008layer_feats_classifier_l2_start0_lambda${lambda}_gamma0.0_${FOLD}"
# path="${output}/${algorithm[i]}/${net}_${classifier}_pretrained_attention_${cohort}aug_minmax_normalize_lr0.0008_mmd_gamma0.0001_${FOLD}"
# path="${output}/${algorithm[i]}/${net}_${classifier}_pretrained_attention_${cohort}aug_minmax_normalize_lr0.0008_${FOLD}"
resume="${path}/model_best.pkl"
# resume="DeepDG/artifacts/model-8hdnycaw:v16/model.ckpt"
# resume="DeepDG/oqx5qczd/checkpoints/epoch=7-step=1400.ckpt"


CUDA_LAUNCH_BLOCKING=1 python train.py --data_dir $data_dir --max_epoch $max_epoch --batch_size $batch_size --net $net --classifier $classifier --task $task --output $output --blocks 4 --attention --pretrained \
--test_envs $test_envs --fold ${FOLD} --dataset $dataset --cohort $cohort --algorithm ${algorithm[i]} --steps_per_epoch ${steps_per_epoch} --augmentation --schuse --schusech cos \
--save_model_every_checkpoint --checkpoint_freq ${checkpoint_freq} --lr 8e-4 --gpu_id ${gpu_ids} --N_WORKERS 1 --num_classes 3 --mmd_gamma 0.0001 --accum_iter ${accum_iter} --groupdro_eta 1 \
--mldg_beta 0.01 --align_start 0 --align_lambda $lambda --align_gamma 0 --align_loss ${loss} --layer classifier --mixupalpha 0.2 --resume=${resume} --eval --write_raw_score #--config=${path}/config.json #--eval --write_raw_score #--save_emb #--replace_att #--config=${path}/config.json #--write_raw_score --eval #--start_epoch 0 #--eval --write_raw_score # # --eval  #--mldg_beta 10


# # Group_DRO
# python train.py --data_dir ~/myexp30609/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output ~/tmp/test00 \
# --test_envs 0 --dataset PACS --algorithm GroupDRO --groupdro_eta 1 

# # ANDMask
# python train.py --data_dir /home/lw/lw/data/PACS/ --max_epoch 3 --net resnet18 --task img_dg --output /home/lw/lw/test00 \
# --test_envs 0 --dataset PACS --algorithm ANDMask --tau 1 