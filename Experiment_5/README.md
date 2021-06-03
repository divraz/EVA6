# Model 1

## Target
- Create a skeleton of model
- Introduce normalization in data augmentation

## Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
         MaxPool2d-2           [-1, 10, 13, 13]               0
            Conv2d-3           [-1, 10, 11, 11]             900
         MaxPool2d-4             [-1, 10, 5, 5]               0
            Conv2d-5             [-1, 10, 5, 5]             900
            Linear-6                   [-1, 10]           2,510
================================================================
Total params: 4,400
Trainable params: 4,400
Non-trainable params: 0
----------------------------------------------------------------
```

## Result
- Parameters : `4,400`
- Best Train Accuracy : `98.56`
- Best Test Accuracy : `98.35` 

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_1.png)

## Analysis
- Model is slightly over-fitting
- We can increase the capacity of the model.

# Model 2
## Target
- Increase the capacity of the model
- Introduce LeakyReLU and Batch Normalization after each layer
- Reduce Learning Rate after every 6 epochs

## Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
         LeakyReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
         MaxPool2d-4           [-1, 16, 13, 13]               0
            Conv2d-5           [-1, 16, 11, 11]           2,304
         LeakyReLU-6           [-1, 16, 11, 11]               0
       BatchNorm2d-7           [-1, 16, 11, 11]              32
         MaxPool2d-8             [-1, 16, 5, 5]               0
            Conv2d-9             [-1, 16, 5, 5]           2,304
        LeakyReLU-10             [-1, 16, 5, 5]               0
      BatchNorm2d-11             [-1, 16, 5, 5]              32
           Linear-12                   [-1, 10]           4,010
================================================================
Total params: 8,858
Trainable params: 8,858
Non-trainable params: 0
----------------------------------------------------------------
```
## Result
- Parameters : `8,858`
- Best Train Accuracy : `99.96`
- Best Test Accuracy : `99.21` 

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_2.png)

## Analysis
- Test Accuracy `99.1` is consistent over last few epochs.
- Model is clearly over-fitting
- Dropout is required
- Observing the wrongly classified images, we can introduce a slight rotation in data augmentation

# Model 3

## Target
- Control the Learning Rate as a inverse factor of test accuracy
- Introduce rotation in data augmentation
- Introduce dropout in all the layers of the model.

## Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
         LeakyReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
         MaxPool2d-5           [-1, 16, 13, 13]               0
            Conv2d-6           [-1, 16, 11, 11]           2,304
         LeakyReLU-7           [-1, 16, 11, 11]               0
       BatchNorm2d-8           [-1, 16, 11, 11]              32
           Dropout-9           [-1, 16, 11, 11]               0
        MaxPool2d-10             [-1, 16, 5, 5]               0
           Conv2d-11             [-1, 16, 5, 5]           2,304
        LeakyReLU-12             [-1, 16, 5, 5]               0
      BatchNorm2d-13             [-1, 16, 5, 5]              32
          Dropout-14             [-1, 16, 5, 5]               0
           Linear-15                   [-1, 10]           4,010
          Dropout-16                   [-1, 10]               0
================================================================
Total params: 8,858
Trainable params: 8,858
Non-trainable params: 0
----------------------------------------------------------------
```

## Result
- Parameters : `8,858`
- Best Train Accuracy : `87.94`
- Best Test Accuracy : `99.42` 

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_3.png)

## Analysis
- Test Accuracy `99.4` in 8th epoch.
- Test Accuracy `99.4` consistent over last 3 epochs.
```
EPOCH: 7 [0.0018]
Loss=0.12311051040887833 Batch_id=468 Accuracy=87.89: 100%|██████████| 469/469 [00:20<00:00, 22.82it/s]
Test set: Average loss: 0.0197, Accuracy: 9934/10000 (99.34%)

EPOCH: 8 [0.00054]
Loss=0.2126765102148056 Batch_id=468 Accuracy=87.94: 100%|██████████| 469/469 [00:20<00:00, 23.14it/s]
Test set: Average loss: 0.0194, Accuracy: 9940/10000 (99.40%)

EPOCH: 9 [4.86e-05]
Loss=0.18105846643447876 Batch_id=468 Accuracy=87.94: 100%|██████████| 469/469 [00:20<00:00, 22.97it/s]
Test set: Average loss: 0.0191, Accuracy: 9942/10000 (99.42%)

EPOCH: 10 [4.374e-06]
Loss=0.14830487966537476 Batch_id=468 Accuracy=87.77: 100%|██████████| 469/469 [00:20<00:00, 23.13it/s]
Test set: Average loss: 0.0192, Accuracy: 9942/10000 (99.42%)
```
- Model is under-fitting.
- We can add more capacity but then we will have to cross 10k parameters
- Seeing the wrong image set, we can conclude that it is difficult even for a human to predict those images correctly.

## Wrongly classified image from test dataset along with predicted class probabilities.
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_wrong.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_wrong_1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_wrong_2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_5/images/5_wrong_3.png)
