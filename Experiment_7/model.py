import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  '''
  4 blocks
  Each block should have:
    - normal convolution with 3x3
    - depthwise seperable convolution with 3x3
    - dilated convolution
    - depthwise seperable convolution with stride of 2
  '''
  def __init__(self):
    super(Net, self).__init__()

    #conv block 1
    #input 32 x 32 x 3
    self.conv_block_1 = nn.Sequential (
        # input = 32 x 32 x 3
        # output = 32 x 32 x 32
        nn.Conv2d (in_channels = 3, 
                   out_channels = 32, 
                   kernel_size = 3,
                   padding = 'same',
                   padding_mode = 'reflect',
                   dilation = 2),
        nn.LeakyReLU (),
        nn.BatchNorm2d (32),
        nn.Dropout (0.2),

        # input = 32 x 32 x 32
        # output = 32 x 32 x 32

        # channel wise convolution
        nn.Conv2d (in_channels = 32,
                   out_channels = 32,
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect',
                   groups = 32),
        # point wise convolution
        nn.Conv2d (in_channels = 32,
                   out_channels = 32,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (32),
        nn.Dropout (0.1),
        
        # input = 32 x 32 x 32
        # output = 15 x 15 x 32
        # channel wise convolution
        nn.Conv2d (in_channels = 32,
                   out_channels = 32,
                   kernel_size = 3,
                   stride = 2,
                   groups = 32),
        # point wise convolution
        nn.Conv2d (in_channels = 32,
                   out_channels = 32,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (32),
        nn.Dropout (0.1)
    )

    #conv block 2
    #input 15 x 15 x 32
    self.conv_block_2 = nn.Sequential (
        # input = 15 x 15 x 32
        # output = 15 x 15 x 64
        nn.Conv2d (in_channels = 32, 
                   out_channels = 64, 
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect'),
        nn.LeakyReLU (),
        nn.BatchNorm2d (64),
        nn.Dropout (0.1),

        # input = 15 x 15 x 64
        # output = 15 x 15 x 64

        # channel wise convolution
        nn.Conv2d (in_channels = 64,
                   out_channels = 64,
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect',
                   groups = 64),
        # point wise convolution
        nn.Conv2d (in_channels = 64,
                   out_channels = 64,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (64),
        nn.Dropout (0.1),

        # input = 15 x 15 x 64
        # output = 7 x 7 x 64
        # channel wise convolution
        nn.Conv2d (in_channels = 64,
                   out_channels = 64,
                   kernel_size = 3,
                   stride = 2,
                   groups = 64),
        # point wise convolution
        nn.Conv2d (in_channels = 64,
                   out_channels = 64,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (64),
        nn.Dropout (0.1),
    )

    #conv block 3
    #input 7 x 7 x 64
    self.conv_block_3 = nn.Sequential (
        # input = 7 x 7 x 64
        # output = 7 x 7 x 128
        nn.Conv2d (in_channels = 64, 
                   out_channels = 128, 
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect'),
        nn.LeakyReLU (),
        nn.BatchNorm2d (128),
        nn.Dropout (0.1),

        # input = 7 x 7 x 128
        # output = 7 x 7 x 128

        # channel wise convolution
        nn.Conv2d (in_channels = 128,
                   out_channels = 128,
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect',
                   groups = 128),
        # point wise convolution
        nn.Conv2d (in_channels = 128,
                   out_channels = 128,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (128),
        nn.Dropout (0.1), 

        # input = 7 x 7 x 128
        # output = 3 x 3 x 128
        # channel wise convolution
        nn.Conv2d (in_channels = 128,
                   out_channels = 128,
                   kernel_size = 3,
                   stride = 2,
                   groups = 128),
        # point wise convolution
        nn.Conv2d (in_channels = 128,
                   out_channels = 128,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (128),
        nn.Dropout (0.1),
    )

    #conv block 4
    #input 3 x 3 x 128
    self.conv_block_4 = nn.Sequential (
        # input = 3 x 3 x 128
        # output = 3 x 3 x 256
        
        nn.Conv2d (in_channels = 128, 
                   out_channels = 128, 
                   kernel_size = 3,
                   padding = 1,
                   padding_mode = 'reflect',
                   groups = 128),
        nn.Conv2d (in_channels = 128,
                   out_channels = 256,
                   kernel_size = 1),
        nn.LeakyReLU (),
        nn.BatchNorm2d (256),
        nn.Dropout (0.1),

        # input = 3 x 3 x 256
        # output = 1 x 1 x 10

        # channel wise convolution
        nn.Conv2d (in_channels = 256,
                   out_channels = 256,
                   kernel_size = 3,
                   groups = 256),
        # point wise convolution
        nn.Conv2d (in_channels = 256,
                   out_channels = 10,
                   kernel_size = 1),
    )

    self.gap = nn.Sequential(
        nn.AvgPool2d (kernel_size = 1)
    )

  def forward (self, x):
    #input = 32 x 32 x 3
    #output = 15 x 15 x 32
    x = self.conv_block_1 (x)
    
    #input = 15 x 15 x 32
    #output = 7 x 7 x 64
    x = self.conv_block_2 (x)
    
    #input = 7 x 7 x 64
    #output = 3 x 3 x 128
    x = self.conv_block_3 (x)
    
    #input = 3 x 3 x 128
    #output = 1 x 1 x 10
    x = self.conv_block_4 (x)
    
    x = self.gap (x)
    x = x.view (-1, 10)
    return F.log_softmax(x, dim=-1)
