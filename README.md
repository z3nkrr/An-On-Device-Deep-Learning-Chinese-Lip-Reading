# An On-Device Deep-Learning Chinese Lip-Reading Input Method for Accessible Mobile Text Entry
## Content

- [Introduction](#Introduction)
- [Usage](#Usage)


## Introduction

In this repository, we provide an end-to-end deep lip-reading pipeline as well as training configurations and TFLite edge-deployment utilities. We evaluate our pipeline on the DMCLR Dataset. We obtain **87.83%** accuracy on the held-out Mandarin dataset split. The results demonstrate that our proposed mobile-friendly architecture is highly feasible for consumer hardware deployment. **Especially, we achieve an on-device inference latency of ~180 ms per word segment on a commodity smartphone, validating that local, privacy-preserving lip-reading is fully practical for real-world text entry.**

## Usage

1.Model training:
```
python train.py --data "path/to/DMCLR_Dataset" 
                --batch-size 8 
                --epochs 100 
                --lr 3e-4 
                --save-dir ./checkpoints
```
Key arguments:
- `data`: Path to the dataset root folder containing train and test splits.  
- `batch-size`: Set to 8 by default.  
- `epochs`: Training limit set to 100 epochs with Cosine Annealing learning rate scheduling and early stopping.  
- `lr`: Base learning rate initialized at 3e-4.

Outputs:
- `checkpoints/latest.pt`: State dictionary of the most recent epoch.
- `checkpoints/best.pt`: State dictionary capturing the top validation accuracy.
- `checkpoints/result.csv`: CSV log storing epoch number, training metrics, validation loss, validation accuracy, and learning rates

2.Model Export & Deployment:
Export PyTorch to ONNX:
```
python export_onnx.py --ckpt checkpoints/best.pt 
                      --out checkpoints/model.onnx 
                      --seq-len 40 
                      --spatial-size 88 
                      --in-channels 1 
                      --opset 13
```
Export ONNX to TFLite:
```
python export_tflite.py --ckpt checkpoints/best.pt 
                        --out checkpoints/model.tflite 
                        --seq-len 40 
                        --spatial-size 88
```