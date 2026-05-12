# TEDA

This is the official implementation of the paper "A Topology-Enhanced Dual-stream Attention Model for Parasternal Long-Axis View Identification in Echocardiography". 

## Introduction

The echocardiographic parasternal long-axis (PLAX) view serves as a fundamental reference for the comprehensive evaluation of the heart and its valves. However, subtle inter-view differences arising from the heart’s complex anatomical structure, coupled with inherent speckle noise, pose substantial challenges to accurate PLAX view identification. In this study, we propose a Topology-Enhanced Dual-stream Attention model (TEDA) for PLAX view identification, which exploits the intrinsic topological relationships of cardiac anatomy to improve recognition accuracy and diagnostic reliability. Specifically, we construct an anatomy-guided graph in which major cardiac regions are represented as nodes, and their anatomical and functional dependencies are encoded as weighted edges to reflect the corresponding association strength. Building on this structured representation, a novel dual-stream feature consolidation module integrates spatial and structural cues, followed by a bidirectional attention mechanism that enables interactive fusion between semantic and topological features. This design facilitates the extraction of discriminative representations that accurately capture the anatomical and functional characteristics of the PLAX view. Extensive experiments on our in-house echocardiographic dataset demonstrate that TEDA consistently surpasses state-of-the-art methods, achieving an accuracy of 90.30\%, highlighting its potential for echocardiographic navigation and view localization.![fig2](https://github.com/xielaobanyy/TEDA/blob/main/asset/fig2.png?raw=true)


## Using the code:

The code is stable while using Python 3.9.13, CUDA >=11.6

- Clone this repository:
```bash
git clone https://github.com/xielaobanyy/TEDA
cd TEDA
```

To install all the dependencies :

```bash
conda env create TEDA python==3.9.13
conda activate TEDA
pip install -r requirements.txt
```

## Training and Validation

1. Train the model.
```
python train.py
```
