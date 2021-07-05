import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
  def __init__ (self):
    super (ResNet, self).__init__()

    self.prep = nn.Sequential (
        nn.Conv2d (
            in_channels = 3,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1,
        ),
        nn.BatchNorm2d (
            num_features = 64,
            momentum = 0.9,
            eps = 1e-5
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()
    )

    self.layer_1_x = nn.Sequential (
        nn.Conv2d (
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.MaxPool2d (
            kernel_size = 2
        ),
        nn.BatchNorm2d (
            num_features = 128,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()
    )

    self.layer_1_r = nn.Sequential (
        nn.Conv2d (
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.BatchNorm2d (
            num_features = 128
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU (),
        nn.Conv2d (
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.BatchNorm2d (
            num_features = 128,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()      
    )

    self.layer_2 = nn.Sequential (
        nn.Conv2d (
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.MaxPool2d (
            kernel_size = 2
        ),
        nn.BatchNorm2d (
            num_features = 256,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()
    )

    self.layer_3_x = nn.Sequential (
        nn.Conv2d (
            in_channels = 256,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.MaxPool2d (
            kernel_size = 2
        ),
        nn.BatchNorm2d (
            num_features = 512,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()
    )

    self.layer_3_r = nn.Sequential (
        nn.Conv2d (
            in_channels = 512,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        nn.BatchNorm2d (
            num_features = 512,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU (),
        nn.Conv2d (
            in_channels = 512,
            out_channels = 512,
            kernel_size = 3,
            stride = 1,
            padding = 1
        ),
        
        nn.BatchNorm2d (
            num_features = 512,
            momentum = 0.9
        ),
        nn.Dropout (
            0.05
        ),
        nn.ReLU ()      
    )

    self.mpool = nn.MaxPool2d (
            kernel_size = 4
    )
    self.linear = nn.Linear (
        in_features = 512,
        out_features = 10
    )

  def forward (self, x):
    x = self.prep (x)
    
    y1 = self.layer_1_x (x)
    y2 = self.layer_1_r (y1)
    x = y1 + y2
    
    x = self.layer_2 (x)

    z1 = self.layer_3_x (x)
    z2 = self.layer_3_r (z1)
    x = z1 + z2

    x = self.mpool (x)
    #print (x.shape)
    x = x.view (-1, 512)
    x = self.linear (x)
    #print (x.shape)

    return F.log_softmax (x, dim = -1)
    #return F.softmax (x, dim = -1)
