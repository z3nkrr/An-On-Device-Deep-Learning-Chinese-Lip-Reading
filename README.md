# An On-Device Deep-Learning Chinese Lip-Reading Input Method for Accessible Mobile Text Entry

Reference implementation accompanying the manuscript submitted to
*IEEE Consumer Electronics Magazine*.

## Content

- [Introduction](#introduction)
- [Scope and Limitations](#scope-and-limitations)
- [Usage](#usage)

## Introduction

This repository provides an end-to-end Chinese lip-reading pipeline, training
configurations, and TFLite utilities for edge deployment, together with the
Android application that performs inference, pinyin-dictionary decoding, and
cross-application text injection entirely on the device.

The model is trained and evaluated on the DMCLR dataset. On a held-out split of
DMCLR it reaches **87.83%** word-level accuracy. On a Samsung Galaxy S25 the
exported TFLite model requires approximately **180 ms of model inference per
word segment**. This figure covers model inference only; it excludes camera
capture, frame extraction, landmark detection, mouth cropping, dictionary
decoding, and text injection.

## Scope and Limitations

This repository demonstrates that the complete pipeline — on-device inference,
dictionary decoding, and cross-application text entry — can be integrated and
executed within the resource budget of a consumer smartphone. It does **not**
demonstrate a usable everyday input method.

- The reported accuracy is measured on a held-out split of DMCLR, which was
  recorded under controlled conditions. It does not transfer to live phone
  capture.
- In informal testing with the phone camera, recognition failed even for
  utterances drawn from the training vocabulary. The gap between the dataset's
  recording conditions and live capture is severe rather than marginal, and
  closing it remains open work.
- Word segmentation is semi-automatic: the user taps a button during recording
  to mark word boundaries. The system is therefore not hands-free in its
  current form.
- The latency figure above is model inference only and should not be read as
  end-to-end response time.

We state these limitations explicitly so that this artifact is not mistaken for
a deployable product.

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
