#!/bin/bash
#python translate.py \
#  --model_selection True \
#  --start_step 18000 \
#  --generalize False \
#  --data_dir /apdcephfs/private_yuewan/template_synthesis_dataset/data/template \
#  --intermediate_dir /apdcephfs/private_yuewan/template_synthesis_dataset/intermediate \
#  --checkpoint_dir /apdcephfs/private_yuewan/template_synthesis_dataset/checkpoint_wnoise \
#  --device cuda

python translate.py \
  --data_split tst \
  --batch_size_val 8 \
  --generalize False \
  --data_dir /apdcephfs/private_yuewan/template_synthesis_dataset/data/template \
  --intermediate_dir /apdcephfs/private_yuewan/template_synthesis_dataset/intermediate \
  --checkpoint_dir /apdcephfs/private_yuewan/template_synthesis_dataset/checkpoint_wnoise \
  --checkpoint model_40000_wz.pt # lvgp best
  #--checkpoint model_52000_wz.pt # generalize best
  #--checkpoint model_128000_wz.pt # generalize 2nd best