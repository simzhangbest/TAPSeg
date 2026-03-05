# TAPSeg: An Open-Source Deep Learning Tool for Instance-Level Tooth and Pulp Segmentation in CBCT

## Introduction

**TAPSeg is a deep learning-based tool designed for instance-level segmentation of teeth and dental pulp in Cone Beam Computed Tomography (CBCT) images. This open-source project aims to facilitate research and clinical applications in dental imaging analysis.**

## Availability

**Note:** The source code and pre-trained model weights will be partially released upon acceptance of our paper. We appreciate your patience and interest in our work. The released materials will include:

* **Core model architecture and implementation**
* **Pre-trained weights for tooth and pulp segmentation**
* **Inference scripts and documentation**

## Video Tutorial

**We provide a comprehensive video demonstration explaining the workflow and usage of TAPSeg:**

**`<!-- Add your video link here -->`**

[Video Tutorial](https://www.bilibili.com/video/BV1SCPyziEcG/)*(https://www.bilibili.com/video/BV1SCPyziEcG/)*

### Workflow Overview

**The TAPSeg pipeline consists of the following steps:**

1. **Data Preparation**
   * **Input: CBCT scans in NIfTI (.nii.gz) format.**
2. **Model Inference**
   * **Load pre-trained TAPSeg model**
   * **Run instance-level segmentation on CBCT volumes**
   * **Outputs generated:**
     * **Whole tooth instance masks**
     * **Dental pulp masks**
     * **Hard tissue masks (automatically derived by subtracting the pulp from the whole tooth)**

## Getting Started

**Detailed installation and usage instructions will be provided upon release.**

### Requirements

* **Python 3.8+**
* **PyTorch 1.10+**
* **Additional dependencies (to be specified in requirements.txt)**

## Citation

**Our paper, "TAPSeg: An Open-Source Deep Learning Tool for Instance-Level Tooth and Pulp Segmentation in CBCT", is currently under review. The official citation format will be provided here immediately upon acceptance and publication.**

## License

**License information will be provided upon release.**

## Contact

**For questions and inquiries, please open an issue on this repository.**

## Acknowledgments

**We thank all contributors and the open-source community for their support.**
