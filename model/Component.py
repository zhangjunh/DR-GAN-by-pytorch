import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms

def Tensor2Image(img):
    """
    input (FloatTensor)
    output (PIL.Image)
    """
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def one_hot(label, depth):
    """
    Return the one_hot vector of the label given the depth.
    Args:
        label (LongTensor): shape(batchsize)
        depth (int): the sum of the labels

    output: (FloatTensor): shape(batchsize x depth) the label indicates the index in the output

    >>> label = torch.LongTensor([0, 0, 1])
    >>> one_hot(label, 2)
    <BLANKLINE>
     1  0
     1  0
     0  1
    [torch.FloatTensor of size 3x2]
    <BLANKLINE>
    """
    out_tensor = torch.zeros(len(label), depth)
    for i, index in enumerate(label):
        out_tensor[i][index] = 1
    return out_tensor

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class conv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 3, 96, 96))

    >>> net = conv_unit(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])

    >>> net = conv_unit(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Fconv_unit(nn.Module):
    """The base unit used in the network.

    >>> input = Variable(torch.randn(4, 64, 48, 48))

    >>> net = Fconv_unit(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])

    >>> net = Fconv_unit(64, 16, unsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Decoder(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_z (int): The dimensions of the noise

    >>> Dec = Decoder()
    >>> input = Variable(torch.randn(4, 372))
    >>> output = Dec(input)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_p=2, N_z=50):
        super(Decoder, self).__init__()
        Fconv_layers = [
            Fconv_unit(320, 160),                   #Bx160x6x6
            Fconv_unit(160, 256),                   #Bx256x6x6
            Fconv_unit(256, 256, unsampling=True),  #Bx256x12x12
            Fconv_unit(256, 128),                   #Bx128x12x12
            Fconv_unit(128, 192),                   #Bx192x12x12
            Fconv_unit(192, 192, unsampling=True),  #Bx192x24x24
            Fconv_unit(192, 96),                    #Bx96x24x24
            Fconv_unit(96, 128),                    #Bx128x24x24
            Fconv_unit(128, 128, unsampling=True),  #Bx128x48x48
            Fconv_unit(128, 64),                    #Bx64x48x48
            Fconv_unit(64, 64),                     #Bx64x48x48
            Fconv_unit(64, 64, unsampling=True),    #Bx64x96x96
            Fconv_unit(64, 32),                     #Bx32x96x96
            Fconv_unit(32, 3)                       #Bx3x96x96
        ]

        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.fc = nn.Linear(320+N_p+N_z, 320*6*6)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 320, 6, 6)
        x = self.Fconv_layers(x)
        return x

class Multi_Encoder(nn.Module):
    """
    The multi version of the Encoder.

    >>> Enc = Multi_Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    def __init__(self):
        super(Multi_Encoder, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 321),                #Bx321x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx321x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 321)
        t = x[:, :320]
        w = x[:, 320]
        batchsize = len(w)
        r = Variable(torch.zeros(t.size())).type_as(t)
        for i in range(batchsize):
            r[i] = t[i] * w[i]
        r = torch.sum(r, 0, keepdim=True).div(torch.sum(w))

        return torch.cat((t,r.type_as(t)), 0)

class Encoder(nn.Module):
    """
    The single version of the Encoder.

    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    def __init__(self):
        super(Encoder, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 320),                #Bx320x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx320x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        return x


class Generator(nn.Module):
    """
    >>> G = Generator()

    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> pose = Variable(torch.randn(4, 2))
    >>> noise = Variable(torch.randn(4, 50))

    >>> output = G(input, pose, noise)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_p=2, N_z=50, single=False):
        super(Generator, self).__init__()
        if single:
            self.enc = Encoder()
        else:
            self.enc = Multi_Encoder()
        self.dec = Decoder(N_p, N_z)

    def forward(self, input, pose, noise):
        x = self.enc(input)
        # print('{0}/t{1}/t{2}'.format(x.size(), pose.size(), noise.size()))
        x = torch.cat((x, pose, noise), 1)
        x = self.dec(x)
        return x

class Discriminator(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_d (int): The sum of the identities

    >>> D = Discriminator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = D(input)
    >>> output.size()
    torch.Size([4, 503])
    """
    def __init__(self, N_p=2, N_d=500):
        super(Discriminator, self).__init__()
        conv_layers = [
            conv_unit(3, 32),                   #Bx32x96x96
            conv_unit(32, 64),                  #Bx64x96x96
            conv_unit(64, 64, pooling=True),    #Bx64x48x48
            conv_unit(64, 64),                  #Bx64x48x48
            conv_unit(64, 128),                 #Bx128x48x48
            conv_unit(128, 128, pooling=True),  #Bx128x24x24
            conv_unit(128, 96),                 #Bx96x24x24
            conv_unit(96, 192),                 #Bx192x24x24
            conv_unit(192, 192, pooling=True),  #Bx192x12x12
            conv_unit(192, 128),                #Bx128x12x12
            conv_unit(128, 256),                #Bx256x12x12
            conv_unit(256, 256, pooling=True),  #Bx256x6x6
            conv_unit(256, 160),                #Bx160x6x6
            conv_unit(160, 320),                #Bx320x6x6
            nn.AvgPool2d(kernel_size=6)         #Bx320x1x1
        ]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(320, N_d+N_p+1)

    def forward(self,input):
        x = self.conv_layers(input)
        x = x.view(-1, 320)
        x = self.fc(x)
        return x
