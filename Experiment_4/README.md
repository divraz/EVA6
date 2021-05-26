# Back Propogation

## Training model on Excel
![Test PDF 1](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/back_prop.png)

## Graphs for varying learning rates
![Test Image 1](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/Learning%20Rate%20Variation.png)

# Model Training

## Model Definition
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
            Conv2d-3           [-1, 32, 24, 24]           4,640
       BatchNorm2d-4           [-1, 32, 24, 24]              64
            Conv2d-5           [-1, 10, 24, 24]             330
         MaxPool2d-6           [-1, 10, 12, 12]               0
         Dropout2d-7           [-1, 10, 12, 12]               0
            Conv2d-8           [-1, 16, 10, 10]           1,456
       BatchNorm2d-9           [-1, 16, 10, 10]              32
           Conv2d-10             [-1, 16, 8, 8]           2,320
      BatchNorm2d-11             [-1, 16, 8, 8]              32
        Dropout2d-12             [-1, 16, 8, 8]               0
           Conv2d-13             [-1, 16, 6, 6]           2,320
      BatchNorm2d-14             [-1, 16, 6, 6]              32
        Dropout2d-15             [-1, 16, 6, 6]               0
           Conv2d-16             [-1, 32, 4, 4]           4,640
      BatchNorm2d-17             [-1, 32, 4, 4]              64
        AvgPool2d-18             [-1, 32, 1, 1]               0
           Linear-19                   [-1, 10]             330
================================================================
Total params: 16,452
Trainable params: 16,452
Non-trainable params: 0
----------------------------------------------------------------
```
## Model Tricks
- Always use Relu after convolution
- Use Batch Normalization only after Relu
- Use dropout after MaxPool
- Reduce the channel size to 10 from 32 using **(1 x 1) kernel** before MaxPool
- Use Global Average Pooling

## Data Augmentation
- Use Normalization
- Use RandomRotation (+5, -5 degrees)
- Batch size of 128

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/images_mnist.png)

## Training
Training happens for 19 epochs for 2 optimizer functions:

### optimizer = optim.Adam (model.parameters (), lr = 0.003)
Achieves **99.5%** validation accuracy in the **12th** epoch
```
epoch=12 batch_id=468 correct=59417/60032 loss=0.04492028430104256: 100%|██████████| 469/469 [00:33<00:00, 14.04it/s]
Test set: Average loss: 0.0206, Accuracy: 9945/10000 (99.5%)
```
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/adam_loss.png)

### optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
Achieves **99.5%** validation accuracy in the **14th** epoch
```
epoch=14 batch_id=468 correct=59409/60032 loss=0.03672423213720322: 100%|██████████| 469/469 [00:33<00:00, 13.84it/s]
Test set: Average loss: 0.0171, Accuracy: 9947/10000 (99.5%)

```
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/sgd_loss.png)

### sample output
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/classification_mnist.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_4/classification_mnist_2.png)

# Concepts

## How many layers
We need to do convolution till the receptive field should be greater than or equal to the size of the image. So the layers will be calculated accordingly.

## MaxPooling
If image size is larger than what is required to do the classification we reduce the image size by the ratio of 4 by maxpooling. It is generally (2 x 2) kernel with a stride of 2.

## 1x1 Convolutions
These convolutions are used to increase/descrease the size of channels as per the need in the model. Helps in reducing the number of parameters.

## 3x3 Convolutions
Kernel size of 3x3 performs convolution increasing the receptive field by 2.

## Receptive Field
The term receptive field refers to the region of visual space where changes in luminance influence the activity of a single neuron.

## SoftMax
This activation function is used to level the outputs in a probability like numbers.

## Learning Rate
This is the factor by which the delta updates happens in the model parameters. Higher the learning rate, higher the change.

## Kernels and how do we decide the number of kernels?
Generally kernel size is taken (3x3) because thats what is accelerated in nvidia gpus. We decide the number of kernels based on number of channels in current layer and the number of channels requried in next layer.
For example if current layer has 16 channels then (3 x 3 x 16) kernel size will be used. and if the next layer needs to have 32 channels then ((3 x 3 x 16) x 32) kernel size will be used.

## Batch Normalization
All the layers perform convolution and addition so the values in last layers can explode and model will take forever to learn. So each layer is normalized using batch normalization which results in each layer having values in a range.

## Image Normalization
The meand and standard deviation of Training image set is stored and applied to traning data before the training starts. Helps in achieving better accuracy.

## Position of MaxPooling
- After 2-4 layers from beginnning
- Before 4-6 layers from end
- Maybe after every 4 convolutions

## Concept of Transition Layers
To change the number of channels with ease and least parameters.

## DropOut
Randomly drop a percentage of values from the layer to help generalize the model.

## When do we introduce DropOut, or when do we know we have some overfitting
When training accuracy is higher than validation accuracy and training accuracy keeps getting higher whereas validation accuracy is same or decreasing, we know our model is overfitting. to overcome this we introduce dropout to generalize the model.

## How do we know our network is not going well, comparatively, very early
- We can observe the change in training accuracy and validation accuracy. If both are increasing with the same pace that means it is going well otherwise something is wrong.
- Accuracy should usually start with 60% and soon cross 80%, if the model is not in this direction, then something is wrong.

## Batch Size, and effects of batch size
Batch size is the number of data points on which gpu can do simultaneous convolutions. Higher the batch size, better the loss, faster the training, higher the accuracy. 
