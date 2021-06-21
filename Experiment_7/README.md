## Data Augmentation
### Mean and Standard Deviation
First we load the data and find the mean and standard deviation for train set.
```
from EVA6.Experiment_7.get_mean_and_std import get_mean_std
get_mean_std (trainloader)
>>> (tensor([0.4914, 0.4822, 0.4465]), tensor([0.2470, 0.2435, 0.2616]), 391)
```
### Apply Albumentations Transformations
```
train_transform = A.Compose ([
    A.HorizontalFlip (),
    A.ShiftScaleRotate (shift_limit = 0.05, scale_limit = 0.1, rotate_limit = 15, p = 0.5),
    A.CoarseDropout (max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), p=0.4),
    A.Normalize (mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
    ToTensorV2()
])
test_transform = A.Compose ([
    A.Normalize (mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
    ToTensorV2()
])
```
### Visualize the augmented data
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/aug1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/aug2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/aug3.png)

## Model
Use depthwise seperable convolutions
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             896
         LeakyReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]             320
            Conv2d-6           [-1, 32, 32, 32]           1,056
         LeakyReLU-7           [-1, 32, 32, 32]               0
       BatchNorm2d-8           [-1, 32, 32, 32]              64
           Dropout-9           [-1, 32, 32, 32]               0
           Conv2d-10           [-1, 32, 15, 15]             320
           Conv2d-11           [-1, 32, 15, 15]           1,056
        LeakyReLU-12           [-1, 32, 15, 15]               0
      BatchNorm2d-13           [-1, 32, 15, 15]              64
          Dropout-14           [-1, 32, 15, 15]               0
           Conv2d-15           [-1, 64, 15, 15]          18,496
        LeakyReLU-16           [-1, 64, 15, 15]               0
      BatchNorm2d-17           [-1, 64, 15, 15]             128
          Dropout-18           [-1, 64, 15, 15]               0
           Conv2d-19           [-1, 64, 15, 15]             640
           Conv2d-20           [-1, 64, 15, 15]           4,160
        LeakyReLU-21           [-1, 64, 15, 15]               0
      BatchNorm2d-22           [-1, 64, 15, 15]             128
          Dropout-23           [-1, 64, 15, 15]               0
           Conv2d-24             [-1, 64, 7, 7]             640
           Conv2d-25             [-1, 64, 7, 7]           4,160
        LeakyReLU-26             [-1, 64, 7, 7]               0
      BatchNorm2d-27             [-1, 64, 7, 7]             128
          Dropout-28             [-1, 64, 7, 7]               0
           Conv2d-29            [-1, 128, 7, 7]          73,856
        LeakyReLU-30            [-1, 128, 7, 7]               0
      BatchNorm2d-31            [-1, 128, 7, 7]             256
          Dropout-32            [-1, 128, 7, 7]               0
           Conv2d-33            [-1, 128, 7, 7]           1,280
           Conv2d-34            [-1, 128, 7, 7]          16,512
        LeakyReLU-35            [-1, 128, 7, 7]               0
      BatchNorm2d-36            [-1, 128, 7, 7]             256
          Dropout-37            [-1, 128, 7, 7]               0
           Conv2d-38            [-1, 128, 3, 3]           1,280
           Conv2d-39            [-1, 128, 3, 3]          16,512
        LeakyReLU-40            [-1, 128, 3, 3]               0
      BatchNorm2d-41            [-1, 128, 3, 3]             256
          Dropout-42            [-1, 128, 3, 3]               0
           Conv2d-43            [-1, 128, 3, 3]           1,280
           Conv2d-44            [-1, 256, 3, 3]          33,024
        LeakyReLU-45            [-1, 256, 3, 3]               0
      BatchNorm2d-46            [-1, 256, 3, 3]             512
          Dropout-47            [-1, 256, 3, 3]               0
           Conv2d-48            [-1, 256, 1, 1]           2,560
           Conv2d-49             [-1, 10, 1, 1]           2,570
        AvgPool2d-50             [-1, 10, 1, 1]               0
================================================================
Total params: 182,474
Trainable params: 182,474
Non-trainable params: 0
----------------------------------------------------------------
```

## Training

### OneCycle Learning Rate policy
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/learning_rate.png)

### Training Logs
for last 6 epochs
```
EPOCH: 34 [0.04359935274376268]
Loss=0.7262353301048279 Batch_id=390 Accuracy=80.41: 100%|██████████| 391/391 [00:10<00:00, 35.77it/s]
Test set: Average loss: 0.0034, Accuracy: 8547/10000 (85.47%)

EPOCH: 35 [0.03062611068504198]
Loss=0.6374956965446472 Batch_id=390 Accuracy=80.77: 100%|██████████| 391/391 [00:10<00:00, 35.70it/s]
Test set: Average loss: 0.0034, Accuracy: 8548/10000 (85.48%)

EPOCH: 36 [0.01978285361794066]
Loss=0.6033517718315125 Batch_id=390 Accuracy=81.15: 100%|██████████| 391/391 [00:10<00:00, 35.99it/s]
Test set: Average loss: 0.0033, Accuracy: 8571/10000 (85.71%)

EPOCH: 37 [0.011205941791481858]
Loss=0.7381899356842041 Batch_id=390 Accuracy=81.35: 100%|██████████| 391/391 [00:11<00:00, 35.24it/s]
Test set: Average loss: 0.0032, Accuracy: 8598/10000 (85.98%)

EPOCH: 38 [0.0050032348483234415]
Loss=0.6880630254745483 Batch_id=390 Accuracy=81.48: 100%|██████████| 391/391 [00:10<00:00, 35.65it/s]
Test set: Average loss: 0.0033, Accuracy: 8598/10000 (85.98%)

EPOCH: 39 [0.0012527354271700075]
Loss=0.64747154712677 Batch_id=390 Accuracy=81.50: 100%|██████████| 391/391 [00:11<00:00, 35.42it/s]
Test set: Average loss: 0.0032, Accuracy: 8592/10000 (85.92%)
```

### Loss and Accuracy Graphs
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/graph.png)

## Inference for wrong classification
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong3.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong4.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong5.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_7/images/wrong6.png)

## Accuracy
### 86%
### Class wise Accuracy
- Accuracy of plane : 86 %
- Accuracy of   car : 94 %
- Accuracy of  bird : 82 %
- Accuracy of   cat : 70 %
- Accuracy of  deer : 85 %
- Accuracy of   dog : 76 %
- Accuracy of  frog : 91 %
- Accuracy of horse : 87 %
- Accuracy of  ship : 94 %
- Accuracy of truck : 91 %
