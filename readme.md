# Organ Calculator: CT and MRI Image Segmentation

This repository contains the implementation and research progress for the "Organ Calculator" project. The goal of this project is to process CT and MRI scans using advanced image segmentation techniques to extract valuable anatomical information.

## Features
- **Data Preprocessing**: Efficient organization of the TotalSegmentator dataset with metadata creation and standardization for training.
- **Model Training**: Utilizes the nnUNet framework for state-of-the-art segmentation model training.
- **Anatomical Analysis**: Developed functions to compute anatomical ratios (e.g., paraspinal muscle left-to-right ratios for scoliosis prediction).
- **Future Directions**:
  - Developing CycleGAN models to synthesize MRI scans from CT scans.
  - Fine-tuning nnUNet for multimodal datasets to enhance cross-modality performance.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/jannikobenhoff/organ_calculator.git
   cd organ_calculator
