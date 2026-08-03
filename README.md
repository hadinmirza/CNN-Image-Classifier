🦁 Animal Image Classification using a Custom Convolutional Neural Network (CNN)
📌 Project Overview

This project is an end-to-end Deep Learning image classification system that identifies one of five animal species from an input image using a custom-built Convolutional Neural Network (CNN).

The project follows the complete Deep Learning workflow, beginning with dataset selection and analysis, followed by preprocessing, data augmentation, CNN design, model training, hyperparameter tuning, evaluation, and finally deployment as a web application using Flask.

The project was developed as part of an AI/ML learning assignment to demonstrate practical implementation of computer vision and deep learning concepts.

🎯 Problem Statement

Image classification is one of the most common computer vision tasks. The goal of this project is to build a deep learning model capable of automatically recognizing an animal from an uploaded image.

The classifier predicts one of the following five classes:

🐴 Horse
🐵 Monkey
🐯 Tiger
🦌 Deer
🐘 Elephant
❓Business Question

Can a custom Convolutional Neural Network accurately classify different animal species from images?

📊 Problem Type

Multi-Class Image Classification

This is a supervised deep learning classification problem where the model predicts one of five animal categories.

📂 Dataset Information

Source: Kaggle

The dataset was organized into:

train/
validation/
test/
Classes
Horse
Monkey
Tiger
Deer
Elephant
Training Images
Class	Images
Horse	845
Monkey	228
Tiger	345
Deer	581
Elephant	639

The dataset contains images with varying resolutions and aspect ratios, requiring preprocessing before model training.

🛠 Project Workflow
1. Data Loading

The dataset was loaded into Google Colab using Google Drive.

The project verified:

Folder structure
Number of classes
Number of images
Image formats
Image resolutions
2. Dataset Analysis

The dataset was explored before training by examining:

Images per class
Class imbalance
Image resolutions
Dataset structure
Sample images

This helped understand the quality and characteristics of the dataset.

3. Data Preprocessing

All images were preprocessed before training.

The preprocessing pipeline included:

Resizing images to 224 × 224
Converting images to tensors
Pixel normalization using ImageNet mean and standard deviation

Normalization improves training stability and convergence.

4. Data Augmentation

To improve generalization and reduce overfitting, augmentation was applied only to the training dataset.

Augmentations included:

Random horizontal flip
Random rotation
Random resized crop
Color jitter
Random affine transformation

Validation and test datasets were only resized and normalized.

5. Dataset Preparation

The processed datasets were loaded using:

ImageFolder
DataLoader

Separate loaders were created for:

Training
Validation
Testing

Mini-batches were used during training to improve efficiency.

6. Custom CNN Architecture

A custom Convolutional Neural Network was implemented using PyTorch.

Architecture:

3 Convolutional Layers
ReLU Activation
Max Pooling
Fully Connected Layer
Dropout Layer
Output Layer (5 Classes)

Unlike transfer learning, the CNN was built completely from scratch.

7. Model Training

The model was trained using:

CrossEntropyLoss
Adam Optimizer
Learning Rate = 0.001
Batch Size = 32

The model was trained over multiple epochs while monitoring:

Training Loss
Validation Loss
Training Accuracy
Validation Accuracy

The best-performing model based on validation accuracy was automatically saved.

8. Hyperparameter Tuning

Model performance was improved through additional training and parameter tuning.

Experiments included:

Increasing training epochs
Adjusting learning behavior
Saving the best-performing checkpoint

The tuned model achieved better generalization than the initial model.

9. Model Evaluation

The final model was evaluated using the unseen test dataset.

Evaluation Metrics
Metric	Value
Accuracy	80.43%
Precision	0.8063
Recall	0.8043
F1 Score	0.8019

Additional evaluation included:

Confusion Matrix
Classification Report
Training vs Validation Accuracy Curve
📈 Results

The custom CNN successfully learned meaningful visual features and achieved approximately 80% test accuracy on unseen images.

Performance Summary:

✅ Test Accuracy: 80.43%
✅ Precision: 0.8063
✅ Recall: 0.8043
✅ F1 Score: 0.8019

The model performed particularly well on Horse and Elephant images, while Deer classification remained comparatively more challenging due to fewer samples and visual similarity with other classes.

🌐 Web Application

A simple web application was developed using Flask to demonstrate the trained model.

Features:

Upload an animal image
Automatic image preprocessing
Load trained CNN model
Predict animal class
Display prediction on the webpage

This converts the trained deep learning model into an interactive application.

📁 Project Structure
Animal-Image-Classifier
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   └── checkpoint.pth
│
├── notebooks/
│   └── Animal_Image_Classifier.ipynb
│
├── static/
│   ├── css/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
💻 Technologies Used
Python
PyTorch
Torchvision
NumPy
Matplotlib
Pillow (PIL)
Flask
HTML
CSS
Google Colab
Git
GitHub
📚 What I Learned

During this project I gained practical experience with:

Image classification
Computer Vision fundamentals
Convolutional Neural Networks (CNNs)
Data preprocessing
Data augmentation
PyTorch model development
Model training
Hyperparameter tuning
Model evaluation
Saving and loading trained models
Building Flask web applications
Deploying a trained deep learning model
Git and GitHub workflow
⚠ Challenges Faced

Some challenges encountered during this project included:

Selecting and analyzing an appropriate image dataset.
Handling varying image sizes and resolutions.
Designing a custom CNN architecture from scratch.
Preventing overfitting through data augmentation.
Improving model performance through hyperparameter tuning.
Loading trained model checkpoints correctly into the Flask application.
Ensuring preprocessing during inference matched the preprocessing used during training.
Testing the model on unseen real-world images, where performance varied compared to the test dataset.
🔮 Future Improvements

Possible improvements include:

Training on a larger dataset.
Collecting more balanced data for underrepresented classes.
Applying advanced hyperparameter tuning.
Using transfer learning models such as ResNet or EfficientNet.
Deploying the application online using Render or Hugging Face Spaces.
Adding prediction confidence scores and top-k predictions.
📝 Project Summary
What this repository contains

A complete end-to-end Deep Learning project for animal image classification using a custom-built CNN and a Flask web application.

What was completed
Dataset analysis
Data preprocessing
Data augmentation
Custom CNN architecture
Model training
Hyperparameter tuning
Model evaluation
Flask web application
Git and GitHub version control

📅 Project Implementation Timeline
🗓 Day 1 — Data Preparation & Baseline CNN

The first day focused on understanding the dataset and building the first working CNN model.

Tasks Completed
Selected the animal image dataset
Loaded dataset into Google Colab
Performed dataset analysis
Explored class distribution and image resolutions
Applied image preprocessing
Implemented data augmentation
Created DataLoaders using ImageFolder
Designed a custom CNN architecture from scratch
Trained the baseline CNN model
Saved the initial trained model
🗓 Day 2 — Model Improvement & Evaluation

The second day focused on improving model performance and evaluating the trained CNN.

Tasks Completed
Studied popular CNN architectures (LeNet, AlexNet, VGG, ResNet) for inspiration
Improved the custom CNN architecture
Performed hyperparameter tuning
Trained the improved CNN model
Saved the best-performing model checkpoint
Evaluated the model on the unseen test dataset
Generated:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report
Compared baseline and improved model performance
Finalized the model for deployment
🗓 Day 3 — Model Deployment & Web Application

The final day focused on deploying the trained model as a simple web application.

Tasks Completed
Loaded the trained CNN model for inference
Stored the trained model in the project directory
Built a Flask backend
Designed the frontend using HTML and CSS
Implemented image upload functionality
Added image preprocessing for inference
Connected the Flask application with the trained CNN
Displayed prediction results on the webpage
Tested the application using both test images and unseen internet images
Finalized the project structure and GitHub repository


👨‍💻 Author

Muhammad Hadin Mirza

Computer Engineering Student

Machine Learning Enthusiast

This project was developed for learning purposes and to demonstrate an end-to-end Deep Learning workflow using PyTorch and Flask as part of an academic portfolio.