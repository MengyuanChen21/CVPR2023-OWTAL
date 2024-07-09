#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--max_seqlen 500 \
--lr 0.00005 \
--k 7 \
--dataset_name Thumos14reduced \
--path_dataset /data_SSD1/cmy/CO2-THUMOS-14 \
--use_model CO2 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2 \
--group_name CELL \
--model_name split1_ckpt \
--split_idx 1 \
--k_edl 7 \
--num_centers 2 \
--seed 0 \
--without_wandb