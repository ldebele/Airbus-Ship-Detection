

## Model Architecture
The U-Net architecture used in this project consists of an encoder (contracting) and decoder (expanding) network structure. The encoder part of the network is a series of convolutional and max-pooling layers that downsample the input images to learns a feature map of the input images, while the decoder part consists of upsampling layers tht reconstruct the image to the orginal resolution to precise location of the object in the image.

## Datasets
The dataset used in this project consists of annotated images. 
you can find the dataset [Dataset]()

## Data Preparation
The data preparation process includes the following steps:
1. Convert the Run-Length-Encoded decode format into binary masked image.
2. Drop 
2. Normalizing and Resizing the images and masks.
3. Splitting the data into training and validation sets in the ratio of 80:20 sets.
4. Converting the data into `tf.data.Dataset` format for efficient processing.

## Training the Model

