
# Pneumonia Detection from Chest X-rays using Deep Learning

This project uses Convolutional Neural Networks (CNNs) to detect Pneumonia from chest X-ray images with high accuracy(94%). Built using TensorFlow and enhanced with Grad-CAM for visual explainability, it aims to assist medical diagnosis using AI.

## Project Highlights

- Model: Custom AlexNet-based CNN architecture
- Accuracy: Achieved over 94% on validation data
- Explainability: Used Grad-CAM to visualize which parts of the X-ray contributed to predictions
- Evaluated using Accuracy, Precision, Recall, Loss graphs

## Dataset

- Source: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Size: ~5,000 images categorized as `NORMAL` and `PNEUMONIA`
- Preprocessing:
  - Image resizing
  - Normalization
  - Data Augmentation (rotation, flipping, etc.)

## Tech Stack

- Language: Python
- Libraries: TensorFlow, NumPy, OpenCV, Matplotlib
- Deep Learning: CNN with AlexNet-inspired architecture
- Explainability: Grad-CAM for visual model interpretation

## Model Training

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Activation: ReLU & Sigmoid
- Epochs: 20+
- Batch Size: 32

Trained model achieves over 94% accuracy and shows strong generalization.

## Results

- Confusion Matrix
- Training & Validation Accuracy Graphs
- Grad-CAM heatmaps on test X-rays
- Precision & Recall > 90%

## Sample Output

| X-Ray Image | Grad-CAM Heatmap |
|-------------|------------------|
| ![sample](samples/xray.jpg) | ![heatmap](samples/gradcam.jpg) |

## Future Scope

- Convert model to TensorFlow Lite for mobile deployment
- Integrate into a real-time diagnostic web/app interface
- Extend to multi-disease classification (e.g., Tuberculosis, COVID)

## Credits

- Dataset: Paul Mooney (Kaggle)
- Built by: Dhruva Srinivasa
- GitHub: [DHRUVA123175](https://github.com/DHRUVA123175)

## License

This project is for educational and research purposes only.
