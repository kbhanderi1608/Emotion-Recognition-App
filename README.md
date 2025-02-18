#EmotionRecognition
---
This repository represents an android application performing recognition of facial emotions on an image.

#The application**
To detect faces on an image the application uses ML Kit. After detection complete the face image area converted into greyscale 48*48 pixel format, each pixel represents as [0, 1] float number. Finally, converted area fed to the TensorFlow Light convolutional neural network model (simple_classifier.tflite). The model provide output that consist of probabilities for each class: angry, disgust, fear, happy, neutral, sad, surprise.


#The hybrid dataset
To train the CNN model there used hybrid dataset composed of the following datasets images:

CK+ (all images except contempt images).
JAFFE (all images).
FER2013 (all images).
RAF-DB (all images but only 205 happy class images).
The resulting hybrid dataset contains 46614 images and has the following data distribution:

All images was converted into the FER2013 images format - greyscale 48*48 pixels.

The convolutional neural network used
To classify facial emotions the application uses trained deep convolutional neural network (simple_classifier.tflite). Each pixel converted from [0, 255] integer number to [0, 1] float number. The neural network has the following structure:


Parameter	Value
min_delata	0.0001
patience	10
optimizer	Adam
learning_rate	0.0001
loss	categorical_crossentropy
batch_size	96
The DNN model trained on hybrid dataset. The dataset was split into two subsets: a train subset (80%) and a test subset (20%).
Normalized confusion matrix:


Metric	Value (Test subset)
Accuracy	0.678
Precision	0.662
F1	0.647
