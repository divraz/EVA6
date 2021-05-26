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
