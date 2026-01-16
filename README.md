🌿 **Plant Disease Detection Using Convolutional Neural Networks (CNN)**

📌 **Project Overview**

This project implements a deep learning–based Plant Disease Detection System using Convolutional Neural Networks (CNNs). The goal is to classify plant leaf images into healthy or diseased categories to support early diagnosis and improve agricultural productivity.

The model is trained on a labeled dataset of plant leaf images and uses advanced preprocessing and CNN feature extraction techniques to achieve high accuracy.

🚀 **Key Features**

- Automated detection of plant diseases from leaf images
- Deep learning model built using CNN architecture
- Visualization of accuracy and loss curves
- Support for multiple disease categories
- Well-structured and modular code for training & testing

🏗️**Project Structure**
```text
📁 Plant-Disease-Detection-CNN
│
├── train_plant_disease_CNN_model.ipynb     # Model training
├── test_plant_disease_CNN_model.ipynb      # Model testing & evaluation
├── requirements.txt                        # Dependencies
├── README.md                               # Documentation
└── dataset/                                # Plant leaf images
```


🧪 **Technologies Used**

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit (optional for deployment)

📂 **Dataset**

The dataset consists of labeled images of healthy and diseased plant leaves.
Images are preprocessed through:
- Resizing
- Normalization
- Data augmentation
This helps improve generalization and reduce overfitting.

⚙️ **Installation & Setup**

1. Clone the Repository
git clone https://github.com/your-username/plant-disease-detection-cnn.git

2. Install Dependencies
pip install -r requirements.txt

🧠 **Model Training**

Run the training notebook:

train_plant_disease_CNN_model.ipynb


The notebook:
- Loads & preprocesses images
- Builds the CNN model
- Trains the model
- Visualizes accuracy and loss

🧪 **Model Testing**

Run the testing notebook:

test_plant_disease_CNN_model.ipynb

This notebook:
- Loads the trained model
- Tests on unseen images
- Displays predictions & evaluation metrics

📊 **Results**

- High accuracy on validation data
- Effective identification of plant diseases
- Good generalization with proper regularization

🌍 **Applications**

- Smart farming
- Crop monitoring systems
- Agricultural automation
- Research and plant pathology

🔮 **Future Enhancements**

- Streamlit or Flask web application
- Mobile application integration
- Real-time detection via camera
- Support for more plant species

👨‍💻 **Author**

Rahul Khadoliya

Python • Deep Learning Enthusiast
