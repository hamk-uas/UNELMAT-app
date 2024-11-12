# Blister Detection System for Metal Surfaces

## Overview
An intelligent system designed to automatically detect and analyze blisters in color-coated metal surfaces. This project implements two distinct approaches: traditional machine vision and deep learning-based detection, demonstrating the superiority of deep learning in surface defect analysis through a user-friendly web interface.

## Two Detection Approaches

### 1. Traditional Machine Vision (`app`)
- Implemented using OpenCV and traditional image processing techniques
- Features image preprocessing, contour detection, and morphological operations
- Provides basic defect measurements and visualization

### 2. Deep Learning Approach (`ml_app`)
- Utilizes YOLOv5 object detection model
- Offers superior detection accuracy and robustness
- Better handles variations in lighting and surface conditions
- Provides comprehensive defect analysis
- Shows improved performance in complex defect pattern recognition

## Key Features
- Choice between traditional and ML-based detection methods
- Real-time processing and analysis
- Result visualization with highlighted defects
- Historical data tracking
- User-friendly web interface for easy access
- Systematic storage and organization of results

## Technology Stack
- Machine Vision: OpenCV
- Deep Learning: YOLOv5, PyTorch
- Backend: Django, Python
- Frontend: HTML, CSS, Bootstrap
- Storage: Django Media System, SQLite

## Data Management
Images and results are systematically stored in user-created folders within Django's media system, allowing easy access and comparison between both detection approaches.

## Project Status
Successfully implemented both approaches, with the ML-based system demonstrating superior performance in blister detection and classification. Complete with image upload, processing, result visualization, and storage capabilities.

## Acknowledgments
- YOLOv5 by Ultralytics
- makesense.ai for dataset annotation
- OpenCV community
- Django web framework

This project demonstrates the evolution from traditional computer vision to modern deep learning approaches in defect detection systems, showcasing significant improvements in accuracy and reliability while maintaining user-friendly interfaces.