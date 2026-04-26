# Lung Segmentation using Deep Learning

## Overview

This project focuses on lung segmentation from chest X-ray images using deep convolutional neural networks. We implement and compare two architectures:

- U-Net
- ResUNet

The models are evaluated on two datasets:
- SHCXR
- JSRT

We analyze both within-dataset performance and cross-dataset generalization.

---

## Key Results

- High segmentation performance (Dice ≈ 0.95+)
- Small performance drop in cross-dataset experiments
- ResUNet slightly outperforms U-Net
- Models generalize reasonably well across datasets

---

## Project Structure

```
LUNG_SEGMENTATION/
│
├── data/                  
│   ├── shcxr/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── jsrt/
│       ├── images/
│       └── masks/
│
├── raw_data/              
│   ├── shcxr/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── jsrt/
│       ├── images/
│       └── masks/
│
├── pred_vis/              
│
├── dataset.py             
├── preprocess.py          
├── train.py               
├── evaluate.py            
├── run_eval.py            
├── pred_vis.py            
│
├── models.py              
├── unet.py                
├── resunet.py             
│
├── utils.py               
├── MyProject.ipynb        
├── MyProject.sh           
│
├── evaluation_results.csv 
├── *.pth                        
```

---

## Models

### U-Net
- Encoder–decoder architecture  
- Skip connections for spatial detail  
- DoubleConv blocks (Conv + BatchNorm + ReLU)

### ResUNet
- Same structure as U-Net  
- Uses residual blocks  
- Improves training stability and performance  

---

## Training Details

- Input size: 256 × 256
- Loss: Dice + Binary Cross Entropy
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-5
- Scheduler: ReduceLROnPlateau
- Epochs: 25
- Batch size: 8

### Data Augmentation
- Random horizontal flip  
- Small random rotations  

---

## Datasets

### SHCXR
Images: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen  
Masks: https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask  

### JSRT
Images & Masks: https://www.kaggle.com/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset  

---

## How to Run

The datasets are **not included** in this repository.

### 1. Download the datasets

- SHCXR Images: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen  
- SHCXR Masks: https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask  
- JSRT Images & Masks: https://www.kaggle.com/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset  

---

### 2. Place the data in the following structure

```
raw_data/
├── shcxr/
│   ├── images/
│   └── masks/
│
└── jsrt/
    ├── images/
    └── masks/
```

Make sure that:
- Images and masks are correctly matched
- File names correspond between images and masks

---

### 3. Run the project

Open the notebook:

```
MyProject.ipynb
```

Run all cells sequentially to reproduce results.

---

## Evaluation

Metrics used:
- Dice Score
- Intersection over Union (IoU)

Experiments:
- SHCXR → SHCXR  
- SHCXR → JSRT  
- JSRT → JSRT  
- JSRT → SHCXR  

---

## Outputs

- evaluation_results.csv → final metrics  
- *.pth → trained models  
- pred_vis/ → prediction visualizations  
- history_*.csv → training curves  

---

## Future Work

- Domain adaptation for better generalization  
- Attention-based architectures  
- Larger and more diverse datasets  
- Multi-class segmentation  
