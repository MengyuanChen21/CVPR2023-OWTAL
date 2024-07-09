#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--max_seqlen 500 \
--lr 0.00005 \
--k 7 \
--dataset_name Thumos14reduced \
--path_dataset /data_SSD1/cmy/CO2-THUMOS-14 \
--use_model CO2 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2 \
--seed 0 \
--test_ckpt ./ckpt/split0_ckpt.pkl \
--split_idx 0 \
--without_wandb \
--topk_test