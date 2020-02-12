import torch
import torch.nn as nn
from anode.models import ODEBlock, AODEBlock
from torchdiffeq import odeint, odeint_adjoint
from layers import AConv2d, ALinear

gpu_boole = torch.cuda.is_available()

class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class AConv2dTime(AConv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(AConv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x, task = 0, round_ = False):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(AConv2dTime, self).forward(t_and_x, task = task, round_ = round_)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.

    Parameters
    ----------

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, img_size, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            out = self.conv1(x)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.non_linearity(out)
            out = self.conv3(out)
        return out


class ConvODENet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.

    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, img_size, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ConvODENet, self).__init__()
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ConvODEFunc(img_size, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = ODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        self.linear_layer = nn.Linear(self.flattened_dim, self.output_dim)

    def forward(self, x, return_features=False, task = 0, round_ = False):
        features = self.odeblock(x)
        pred = self.linear_layer(features.view(features.size(0), -1))
        if return_features:
            return features, pred
        return pred


class AConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.

    Parameters
    ----------

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, img_size, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(AConvODEFunc, self).__init__()
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels += augment_dim
        self.num_filters = num_filters

        if time_dependent:
            self.conv1 = AConv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = AConv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = AConv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = AConv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = AConv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = AConv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()
            
        self.task=0
        self.round_=False

    def forward(self, t, x, task = 0, round_ = False):
        self.update_task(task = task, round_ = round_)
        
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x, task = self.task, round_ = self.round_)
            out = self.non_linearity(out)
            out = self.conv2(t, out, task = self.task, round_ = self.round_)
            out = self.non_linearity(out)
            out = self.conv3(t, out, task = self.task, round_ = self.round_)
        else:
            out = self.conv1(x, task = self.task, round_ = self.round_)
            out = self.non_linearity(out)
            out = self.conv2(out, task = self.task, round_ = self.round_)
            out = self.non_linearity(out)
            out = self.conv3(out, task = self.task, round_ = self.round_)
        return out
    
    def update_task(self, task=0, round_= False):
        self.task = task
        self.round_ = round_


class AConvODENet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.

    Parameters
    ----------
    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, img_size, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(AConvODENet, self).__init__()
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = AConvODEFunc(img_size, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = AODEBlock(odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        self.linear_layer = ALinear(self.flattened_dim, self.output_dim)

    def forward(self, x, return_features=False, task = 0, round_ = False):
        features = self.odeblock(x, task = task, round_ = round_)
        pred = self.linear_layer(features.view(features.size(0), -1), task = task, round_ = round_)
        if return_features:
            return features, pred
        return pred