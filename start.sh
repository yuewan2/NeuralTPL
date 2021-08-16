#!/bin/bash
source ~/.bashrc
conda activate retrosyn
cd /apdcephfs/private_yuewan/template_synthesis

#python train.py \
#  --max_step 500000 \
#  --batch_size_trn 48 \
#  --save_per_step 2000 \
#  --val_per_step 2000 \
#  --report_per_step 50 \
#  --generalize True \
#  --data_dir /apdcephfs/private_yuewan/template_synthesis_dataset/data/template \
#  --intermediate_dir /apdcephfs/private_yuewan/template_synthesis_dataset/intermediate \
#  --checkpoint_dir /apdcephfs/private_yuewan/template_synthesis_dataset/checkpoint_wnoise \
#  # --checkpoint @model_98000_wz.pt

#python translate.py \
#  --data_split tst \
#  --batch_size_val 4 \
#  --generalize True \
#  --data_dir /apdcephfs/private_yuewan/template_synthesis_dataset/data/template \
#  --intermediate_dir /apdcephfs/private_yuewan/template_synthesis_dataset/intermediate \
#  --checkpoint_dir /apdcephfs/private_yuewan/template_synthesis_dataset/checkpoint_wnoise \
#  --checkpoint model_128000_wz.pt

