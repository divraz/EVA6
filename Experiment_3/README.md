# Multi Input and Multi Output Model Training Strategy

## Data Strategy

- Download the MNIST dataset
- Create a new Dataset `addition`
- Iterate over MNIST and for each entry:
  - Input 1 is the MNIST image itself.
  - For Input 2 create a random number in range [0, 9] and convert the random number to one hot encoding.
  - Output 1 is the MNIST output
  - Output 2 is the sum of Output 1 and Input 2
- The final data block for single entry will look like this:
![image1.png](https://raw.githubusercontent.com/divyanshuraj6815/EVA6/main/Experiment_3/experiment_3_data.png)


## Network

- We need to merge 2 inputs and give out 2 outputs.
- Our model will look like this:
![image2.png](https://raw.githubusercontent.com/divyanshuraj6815/EVA6/main/Experiment_3/experiment_3_network.png)
- We have used [mish activation](https://arxiv.org/abs/1908.08681) for merging the 2 inputs.
- It looks like this ![image3.png](https://camo.githubusercontent.com/69bf8e8f70b22901e9431a457d82d7a30a2eb4e5/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f333030302f312a52754e4d345956385a756974644c676b715f4d5055772e706e67)

## Training

- We have used SGD optimizer with a momentum of 0.9
- We have used cross entropy loss for both outputs and then added them to get a final loss.
- First we train for 20 epochs the network path involving Input 1 to Output 1 keeping all other weights constant. (Back propogation happens for only loss 1)
  - Accuracy 1: 99.9 % 
  - Accuracy 2: 3.1 % 
  - loss_1: 0.738
  - loss_2: 2324.328
- Second we train for another 20 epochs the network path involving Input 2 to Output 2 keeping all other weights constant. (Back propogation happens for only loss 2)
  - Accuracy 1: 99.84 %
  - Accuracy 2: 35.745 % 
  - loss_1: 2.678
  - loss_2: 750.663
  - ![image4.png](https://raw.githubusercontent.com/divyanshuraj6815/EVA6/main/Experiment_3/loss_after_40.png)
- Third we train all the weights considering loss = loss_1 + loss_2
  - Accuracy 1: **99.91 %** 
  - Accuracy 2: **97.31 %** 
  - loss_1: 121.706
  - loss_2: 118.513
  - ![image5.png](https://raw.githubusercontent.com/divyanshuraj6815/EVA6/main/Experiment_3/loss_after_60.png)

## Inference

- Here are results for few images and random number:
- ![image6.png](https://raw.githubusercontent.com/divyanshuraj6815/EVA6/main/Experiment_3/inference.png)
