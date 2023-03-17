# MNIST Classification with Residual Neural Network (ResNet)
This repository contains a Residual Neural Network (ResNet) implemented with TensorFlow 2. The model is trained and evaluated on the MNIST dataset, which consists of grayscale images of handwritten digits.

# Requirements
1. Python 3.6 or above
2. TensorFlow 2
3. Numpy
4. Matplotlib
# Dataset
The MNIST dataset is included with the TensorFlow Keras API, so there is no need to download the dataset separately. The dataset consists of 60,000 training images and 10,000 test images of size 28x28 pixels.

# Model Architecture
The ResNet implemented in this repository consists of three Residual Blocks, each composed of three convolutional layers with Batch Normalization and ReLU activation, plus a skip connection. The final output of the model is passed through a Global Average Pooling layer and a Dense layer with softmax activation, which predicts the class probabilities.

The model architecture can be summarized as follows:
_________________________________________________________________
Layer (type)                 Output Shape              Param   
=================================================================
res_block (ResBlock)         (None, 14, 14, 64)        124736    
_________________________________________________________________
res_block_1 (ResBlock)       (None, 7, 7, 256)         1133056   
_________________________________________________________________
res_block_2 (ResBlock)       (None, 4, 4, 512)         3489792   
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 10)                5130      
=================================================================
Total params: 4,748,714
Trainable params: 4,741,258
Non-trainable params: 7,456
_________________________________________________________________
# Training
The model was trained on the MNIST training set using the Adam optimizer and Categorical Crossentropy loss function. The training was run for 10 epochs with a batch size of 128.


# Results
The final model achieved an accuracy of around 99.3% on the MNIST test set, which is a good result for this task.

# Usage
To train the model from scratch, simply run the Resnetlike.ipynb file on colab
