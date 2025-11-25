# Three Cars Approacing within 100m! Enhancing Distant Geometry by Tri-Axis Voxel Scanning for Camera-based Semantic Scene Completion

## News
---
- [2025/11/25] Code release
- [2025/02/27] Our paper has been accepted to CVPR2025!
- [2024/11/25] [**arxiv**](https://arxiv.org/abs/2411.16129) preprint released

## Introduction
Camera-based Semantic Scene Completion (SSC) is gaining attentions in the 3D perception field. However, properties such as perspective and occlusion lead to the underestimation of the geometry in distant regions, posing a critical issue for safety-focused autonomous driving systems. To tackle this, we propose ScanSSC, a novel camera-based SSC model composed of a Scan Module and Scan Loss, both designed to enhance distant scenes by leveraging context from near-viewpoint scenes. The Scan Module uses axis-wise masked attention, where each axis employing a near-to-far cascade masking that enables distant voxels to capture relationships with preceding voxels. In addition, the Scan Loss computes the cross-entropy along each axis between cumulative logits and corresponding class distributions in a near-to-far direction, thereby propagating rich context-aware signals to distant voxels. Leveraging the synergy between these components, ScanSSC achieves state-of-the-art performance, with IoUs of 44.54 and 48.29, and mIoUs of 17.40 and 20.14 on the SemanticKITTI and SSCBench-KITTI-360 benchmarks.

## Method

![overview]()

GeoDepth Pretrained Weight  : https://github.com/Ha-coding-user/ScanSSC/releases/download/v1.0/pretrain_geodepth.pth  
EfficientNet                : https://github.com/Ha-coding-user/ScanSSC/releases/download/v1.0/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth  
ScanSSC Weight              : https://github.com/Ha-coding-user/ScanSSC/releases/download/v1.0/ScanSSC_SemanticKitti.ckpt
