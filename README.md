# LTD
Bridging the gap between cumbersome and light detectors via distillation
## Install
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmdet==2.19.0
  
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmdetectin's codes.
  - Add the mmdet/. in our codes to the mmdet/ in mmdetectin's codes.
  
## Train
```
#single GPU
python tools/train.py configs/all_gfl18-101.py --gpus 1

#multi GPU
bash tools/dist_train.sh configs/all_gfl18-101.py 8
```

## Test

```
#single GPU
python tools/test.py configs/all_gfl18-101.py $new_mmdet_pth --eval bbox

#multi GPU
bash tools/dist_test.sh configs/all_gfl18-101.py $new_mmdet_pth 8 --eval bbox
```

## Trained model and log file
  - Trained model can be finded in [GoogleDrive](https://drive.google.com/file/d/1lPxoFExZC5mUtPZ-JWD9opqcsaMu06T2/view?usp=share_link)
  - Log file can be finded in logs/.
