# CVPR2023-OWTAL
An implementation of a baseline method of OWTAL. However, it is not the complete version yet.

### THUMOS-14 Dataset：
We use the 2048-d features provided by MM 2021 paper: Cross-modal Consensus Network for Weakly Supervised Temporal Action Localization. You can get access of the dataset from [Google Drive](https://drive.google.com/file/d/1SFEsQNLsG8vgBbqx056L9fjA4TzVZQEu/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1nspCSpzgwh5AHpSBPPibrQ?pwd=2dej). The annotations are included within this package.

### Pre-trained models:
Pre-trained models can be downloaded from [Google Disk](https://drive.google.com/file/d/1GjiNATcUdJlFpX6rK0FIik7ma2QO-L5c/view?usp=sharing).
They need to be unzipped and put in the directory './ckpt/'.

### Quick start
To test pre-trained models, run:
   ```
   cd scripts
   bash test_split0/1/2.sh
   ```

To train from scratch, run:
   ```
   cd scripts
   bash train_split0/1/2.sh
   ```
