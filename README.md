# Food Image Classifier

## Overview
An advanced deep learning system for food recognition and classification using state-of-the-art convolutional neural networks. The system can identify 101 different food categories with high accuracy and provides nutrition information and recommendations.

## Features
- **Multi-Model Support**: EfficientNet, ResNet, MobileNet architectures
- **Transfer Learning**: Pre-trained models for fast training and high accuracy
- **Data Augmentation**: Comprehensive image augmentation for robust training
- **Nutrition Integration**: Calorie and nutrition information lookup
- **Recommendation System**: Similar foods and healthier alternatives
- **Batch Processing**: Classify multiple images simultaneously
- **Web Integration**: URL image classification support
- **Model Persistence**: Save and load trained models

## Supported Food Classes (101 Categories)
- **Appetizers**: Bruschetta, Deviled eggs, Edamame, Escargots, Guacamole
- **Main Courses**: Pizza, Hamburger, Steak, Grilled salmon, Chicken curry
- **Desserts**: Chocolate cake, Ice cream, Tiramisu, Cheesecake, Apple pie
- **Asian Cuisine**: Sushi, Ramen, Pad thai, Bibimbap, Dumplings
- **And many more...**

## Requirements
```
tensorflow>=2.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
pillow>=8.0.0
requests>=2.28.0
scikit-learn>=1.0.0
```

## Installation
```bash
# Clone repository
git clone https://github.com/username/food-image-classifier.git
cd food-image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Option 1: Food-101 Dataset
```bash
# Download Food-101 dataset
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xzf food-101.tar.gz

# Organize data
python organize_food101.py
```

### Option 2: Custom Dataset
Organize your food images in this structure:
```
food_dataset/
├── train/
│   ├── pizza/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── burger/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── pizza/
│   └── burger/
└── test/
    ├── pizza/
    └── burger/
```

## Usage

### Basic Usage
```python
from food_image_classifier import FoodImageClassifier

# Initialize classifier
classifier = FoodImageClassifier(img_size=(224, 224))

# Setup data
classifier.create_data_generators('food_dataset/train')

# Build and train model
classifier.build_transfer_learning_model('efficientnet')
history = classifier.train(epochs=
