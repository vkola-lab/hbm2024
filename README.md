# Disease-driven Domain Generalization (D3G) for neuroimaging-based assessment of Alzheimer's disease
<i>Diala Lteif, Sandeep Sreerama, Sarah A. Bargal, Bryan A. Plummer, Rhoda Au, and Vijaya B. Kolachalama, Senior Member, IEEE</i>

## Introduction

This repository contains the implementation of a deep learning framework that uses disease-informed prior knowledge to train generalizable models for the classification of Alzheimer's disease. We train of persons with normal cognition (NC), mild cognitive impairment (MCI), and Alzheimer's disease (AD).

We trained a deep neural network that used model-identified regions of disease relevance to inform model training. We trained a classifier to distinguish persons with normal cognition (NC) from those with mild cognitive impairment (MCI) and Alzheimer's disease (AD) by aligning class-wise attention with a unified visual saliency prior computed offline per class over all training data. We demonstrated that our proposed method competes with state-of-the-art methods with improved correlation with postmortem histology, thus grounding our findings with gold standard evidence and paving a way towards validating DG frameworks.

<img src="FigTable/fig1_framework.png" width="1000" />


### Prerequisites

The tool was developed using the following dependencies:

1. PyTorch (1.13 or greater).
2. TorchIO (0.18 or greater).
3. MONAI (0.8 or greater).
3. NumPy (1.19 or greater).
3. tqdm (4.62 or greater).
4. pandas (1.1 or greater).
4. nibabel (3.2 or greater).
5. nilearn (0.8 or greater).
5. matplotlib (3.3 or greater).
6. shap (0.39 or greater).
7. scikit-learn (0.24 or greater).
8. scipy (1.5.4 or greater).

### Installation
You can clone this repository using the following command:
```bash
git clone https://github.com/vkola-lab/d3g.git
```


## Documentation

You can train, validate, and test your model using the bash scripts we provided under ```codebase/DeepDG/scripts```.

The main script for training and inference is ```codebase/DeepDG/train.py```. 

The source and target domains can be set in the ```img_param_init()``` function in ```codebase/DeepDG/utils/util.py```.
You will see we set our domains for ```dataset='MRI'```. 
You can then control what domain(s) is/are the source by setting the --test_envs flag passed to ```codebase/DeepDG/train.py```.

### Training

Our training pipeline consists of the following steps:

1. Train a baseline model on your source domain:

```bash
test_envs="1 2 3"
cohort="NACC_NC_MCI_AD"
batch_size=2
accum_iter=8
steps_per_epoch=200
FOLD=0

cd codebase/DeepDG

python train.py --data_dir <data_dir> --task mri_dg --max_epoch 60 \
--batch_size ${batch_size} --accum_iter ${accum_iter} \
--num_classes 3 --net unet3d --classifier gap  --output <output_dir> \
--blocks 4 --attention --pretrained --test_envs ${test_envs} \
--fold ${FOLD} --dataset MRI --cohort ${cohort} \
--algorithm ERM --steps_per_epoch ${steps_per_epoch} \
--augmentation --schuse --schusech cos \
--save_model_every_checkpoint --checkpoint_freq 1 --lr 8e-4 --gpu_id 0 --N_WORKERS 3 
```

2. Generate the class-wise SHAP prior knowledge offline:
```bash
test_envs="1 2 3"
cohort="NACC_NC_MCI_AD"
batch_size=2
FOLD=0

cd codebase/DeepDG

python visualize_shap.py --data_dir <data_dir> --task mri_dg --batch_size ${batch_size} \
--net unet3d --classifier gap --attention --num_classes 3 --blocks 4 --attention \
--fold ${FOLD} --dataset MRI --cohort $cohort --algorithm ERM --test_envs ${test_envs} \
--N_WORKERS 4 --lr 8e-4 --gpu_id 0 --layer classifier \
--num_classes 3 --resume=<path/to/checkpoint> --eval
```

3. Run our algorithm ```"XAIalign"```:
```bash
test_envs="1 2 3"
cohort="NACC_NC_MCI_AD"
batch_size=2
accum_iter=8
steps_per_epoch=200
FOLD=0
lambda=5e-05
cd codebase/DeepDG

python train.py --data_dir <data_dir> --task mri_dg --max_epoch 60 \
--batch_size ${batch_size} --accum_iter ${accum_iter} \
--num_classes 3 --net unet3d --classifier gap  --output <output_dir> \
--blocks 4 --attention --pretrained --test_envs ${test_envs} \
--fold ${FOLD} --dataset MRI --cohort ${cohort} \
--algorithm XAIalign --align_start 0 --align_lambda ${lambda} --align_loss l2 \
--steps_per_epoch ${steps_per_epoch} --augmentation --schuse --schusech cos \
--save_model_every_checkpoint --checkpoint_freq 1 --lr 8e-4 --gpu_id 0 --N_WORKERS 3 
```

### Inference
Inference can be run using the same command above (3) but with the additional arguments: 
```python 
--eval --resume=<path/to/checkpoint>
```

### Data visualization

Please find the scripts used for plotting and figure generation under the **codebase/utils/** folder.
