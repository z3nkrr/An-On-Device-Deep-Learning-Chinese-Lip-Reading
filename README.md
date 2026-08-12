# An On-Device Deep-Learning Chinese Lip-Reading Input Method for Accessible Mobile Text Entry

Reference implementation accompanying the manuscript submitted to
*IEEE Consumer Electronics Magazine*.

## Content

- [Introduction](#introduction)
- [Scope and Limitations](#scope-and-limitations)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This repository provides an end-to-end Chinese lip-reading pipeline, training
configurations, and TFLite utilities for edge deployment, together with the
Android application that performs inference, pinyin-dictionary decoding, and
cross-application text injection entirely on the device.

The model is trained and evaluated on the DMCLR dataset. On a held-out split it
reaches **87.8%** syllable-level accuracy. On a Samsung Galaxy S25 the exported
TFLite model requires about **180 ms of model inference per syllable segment**
when measured on its own with the device idle, and about **319 ms** when called
from within the running pipeline. Either figure covers model inference only; it
excludes camera capture, frame extraction, landmark detection, mouth cropping,
dictionary decoding, and text injection. End to end, a single utterance takes
roughly **10 s** on the same device.

## Scope and Limitations

This repository demonstrates that the complete pipeline — on-device inference,
dictionary decoding, and cross-application text entry — can be integrated and
executed within the resource budget of a consumer smartphone. It does **not**
demonstrate a usable everyday input method.

- The held-out split of DMCLR shares speakers with the training set, so the
  87.8% figure is speaker-dependent and does not indicate performance for a
  speaker the model has never seen.
- In testing with live phone capture by a speaker outside the dataset,
  syllable-level Top-1 accuracy was **2.9%** and no sentence was recognized
  correctly.
- Word segmentation is semi-automatic: the user taps a button once per
  character while recording. The system is therefore not hands-free in its
  current form.
- Frame extraction and landmark detection account for about 87% of the
  end-to-end latency; model inference is under 10%.

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

## Acknowledgements

The floating-window user interface is built on
[compose-floating-window](https://github.com/only52607/compose-floating-window)
by only52607, used as an unmodified dependency under the Apache License 2.0.

Facial landmark detection uses MediaPipe, and on-device inference uses
TensorFlow Lite, both by Google under the Apache License 2.0.

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full list of
third-party components and their licenses.

## License

The code in this repository is released under the MIT License.
See [LICENSE](LICENSE) for details.

Third-party dependencies remain under their own licenses; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
