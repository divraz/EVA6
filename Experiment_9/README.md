# Data Augmentation

```
train_transform = A.Compose ([
    A.PadIfNeeded (min_height = 40, min_width = 40, border_mode = cv2.BORDER_CONSTANT, value = (0.4914, 0.4822, 0.4465)),
    A.HorizontalFlip (),
    A.CoarseDropout (max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), p=0.8),
    A.ShiftScaleRotate (shift_limit = 0.05, scale_limit = 0.1, rotate_limit = 9, p = 0.5),
    A.RandomCrop (height = 32, width = 32, always_apply = True),
    A.Normalize (mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)),
    ToTensorV2()
])
```
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug3.png)

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/img1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/img2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/img3.png)

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug1.gif)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug2.gif)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/aug3.gif)

# Model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
           Dropout-3           [-1, 64, 32, 32]               0
              ReLU-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,856
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
           Dropout-8          [-1, 128, 16, 16]               0
              ReLU-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
      BatchNorm2d-11          [-1, 128, 16, 16]             256
          Dropout-12          [-1, 128, 16, 16]               0
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,584
      BatchNorm2d-15          [-1, 128, 16, 16]             256
          Dropout-16          [-1, 128, 16, 16]               0
             ReLU-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         295,168
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
          Dropout-21            [-1, 256, 8, 8]               0
             ReLU-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
          Dropout-26            [-1, 512, 4, 4]               0
             ReLU-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
          Dropout-30            [-1, 512, 4, 4]               0
             ReLU-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
          Dropout-34            [-1, 512, 4, 4]               0
             ReLU-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
```

# LR Finder

![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/lr_finder.png)
Found at 0.003

# Training

## One Cycle Policy
```
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = 5e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr = 0.04, 
    epochs = 24, 
    steps_per_epoch = 98,
    pct_start = 0.15, 
    anneal_strategy = 'linear'
)
```
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/lr.png)

## Last 4 epochs
```
EPOCH: 20 [0.007823257959183677]
Loss=0.23774349689483643 Batch_id=97 Accuracy=91.96: 100%|██████████| 98/98 [01:42<00:00,  1.04s/it]
Test set: Average loss: 0.0006, Accuracy: 9040/10000 (90.40%)

EPOCH: 21 [0.005862481488595438]
Loss=0.26606518030166626 Batch_id=97 Accuracy=92.66: 100%|██████████| 98/98 [01:42<00:00,  1.04s/it]
Test set: Average loss: 0.0006, Accuracy: 9111/10000 (91.11%)

EPOCH: 22 [0.0039017050180071983]
Loss=0.20857861638069153 Batch_id=97 Accuracy=93.30: 100%|██████████| 98/98 [01:43<00:00,  1.05s/it]
Test set: Average loss: 0.0006, Accuracy: 9104/10000 (91.04%)

EPOCH: 23 [0.0019409285474189658]
Loss=0.2540911138057709 Batch_id=97 Accuracy=93.98: 100%|██████████| 98/98 [01:45<00:00,  1.08s/it]
Test set: Average loss: 0.0006, Accuracy: 9123/10000 (91.23%)
```

## Train and Loss Graphs
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/graph.png)

# Inference and HeatMap
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/heatmap.png)

## wrong classification
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/wrong1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/wrong2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/wrong3.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/wrong4.png)

## correct classification
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/correct1.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/correct2.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/correct3.png)
![](https://github.com/divyanshuraj6815/EVA6/blob/main/Experiment_9/images/correct4.png)
