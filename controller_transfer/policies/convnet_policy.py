from .base_policy import BaseTorchPolicy

import numpy as np
import torch
from torch import nn as nn
from kornia.geometry.transform import (warp_affine, warp_perspective,
    get_rotation_matrix2d, get_perspective_transform)
from torchvision.transforms import ColorJitter
import torch.nn.functional as F
from torch.distributions import constraints, Distribution, Independent, Normal
import numpy as np
import controller_transfer.torch.pytorch_util as ptu

class CNN(nn.Module):
    # TODO: remove the FC parts of this code
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=nn.Identity(),
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            image_augmentation=False,
            image_augmentation_padding=4,
            augmentation_type='random_crop',
            feature_norm=False,
            color_jitter=False,
            fc_dropout=0.0,
            fc_dropout_length=0,
            **kwargs
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        assert augmentation_type in {'random_crop', 'warp_affine', 'warp_perspective'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type
        self.image_augmentation = image_augmentation
        self.image_augmentation_padding = image_augmentation_padding
        self.augmentation_type = augmentation_type
        self.fc_dropout = fc_dropout
        self.fc_dropout_length = fc_dropout_length
        self.hidden_sizes = hidden_sizes
        self.init_w = init_w
        self.feature_norm = feature_norm
        self.color_jitter = color_jitter
        if self.color_jitter:
            assert self.input_channels == 3
            self.color_jitter_transform = ColorJitter(
                    brightness=(0.5,1.5), 
                    contrast=(1), 
                    saturation=(0.5,1.5), 
                    hue=(-0.1,0.1))

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                if pool_sizes[i] > 1:
                    self.pool_layers.append(
                        nn.MaxPool2d(
                            kernel_size=pool_sizes[i],
                            stride=pool_strides[i],
                            padding=pool_paddings[i],
                        )
                    )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))

        if self.output_conv_channels:
            self.last_fc = None
        else:
            self.fc_layers, self.fc_norm_layers, self.last_fc = self.initialize_fc_layers(self.hidden_sizes,
                self.output_size, self.conv_output_flat_size, self.added_fc_input_size, init_w)

        if self.image_augmentation:
            if self.augmentation_type == 'random_crop':
                self.augmentation_transform = RandomCrop(
                    input_height, self.image_augmentation_padding, device=ptu.device)
            elif self.augmentation_type == 'warp_perspective':
                self.augmentation_transform = WarpPerspective(
                    input_height)
            elif self.augmentation_type == 'warp_affine':
                self.augmentation_transform = WarpAffine(
                    input_height)

        if self.fc_dropout > 0.0:
            self.fc_dropout_layer = nn.Dropout(self.fc_dropout)


    def initialize_fc_layers(self, hidden_sizes, output_size, conv_output_flat_size, added_fc_input_size, init_w):
        fc_layers = nn.ModuleList()
        fc_norm_layers = nn.ModuleList()


        fc_input_size = conv_output_flat_size
        # used only for injecting input directly into fc layers
        fc_input_size += added_fc_input_size
        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_input_size = hidden_size

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            fc_layers.append(fc_layer)

            if self.fc_normalization_type == 'batch':
                fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                fc_norm_layers.append(nn.LayerNorm(hidden_size))

        self.last_fc_input_dim = fc_input_size
        last_fc = nn.Linear(fc_input_size, output_size)
        last_fc.weight.data.uniform_(-init_w, init_w)
        last_fc.bias.data.uniform_(-init_w, init_w)

        return fc_layers, fc_norm_layers, last_fc

    def forward(self, conv_input, extra_fc_input=None, return_last_activations=False, intermediate_output_layer=-1):
        h = conv_input

        if self.color_jitter:
            h = self.color_jitter_transform(h)

        if h.shape[0] > 1 and self.image_augmentation:
            # h.shape[0] > 1 ensures we apply this only during training
            h = self.augmentation_transform(h)

        h = self.apply_forward_conv(h)
        
        if self.feature_norm:
            h = h / (torch.norm(h, dim=(1, 2, 3), keepdim=True).detach())

        if self.output_conv_channels:
            return h
       
        # flatten channels for fc layers
        h = h.flatten(1, 3)
        if self.added_fc_input_size != 0:
            h = torch.cat((extra_fc_input, h), dim=+1)
        h = self.apply_forward_fc(h, intermediate_output_layer)

        if intermediate_output_layer != -1:
            h, intermediate_output = h

        if intermediate_output_layer == -1:
            if return_last_activations:
                return h
            return self.output_activation(self.last_fc(h))
        if return_last_activations:
            return h, intermediate_output
        return self.output_activation(self.last_fc(h)), intermediate_output

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none' and len(self.pool_layers) > i:
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h, intermediate_output_layer=-1):
        if self.fc_dropout > 0.0 and self.fc_dropout_length > 0:
            dropout_input = h.narrow(
                start=0,
                length=self.fc_dropout_length,
                dim=1,
            )
            dropout_output = self.fc_dropout_layer(dropout_input)

            remaining_input = h.narrow(
                start=self.fc_dropout_length,
                length=self.conv_output_flat_size + self.added_fc_input_size - self.fc_dropout_length,
                dim=1)
            h = torch.cat((dropout_output, remaining_input), dim=1)

        if self.added_fc_input_size != 0 and self.fc_insertion_position == 0:
            h = torch.cat((extra_fc_input, h), dim=+1)

        intermediate_output = None
        for i, layer in enumerate(self.fc_layers):
            if i == intermediate_output_layer:
                intermediate_output = h
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)

        if intermediate_output is None:
            return h
        return h, intermediate_output


class GaussianCNNPolicy(CNN):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self.lr = kwargs['lr']
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr
        )


    def forward(self, obs, extra_fc_input=None, intermediate_output_layer=-1):
        if intermediate_output_layer == -1:
            h = super().forward(obs, extra_fc_input=extra_fc_input, return_last_activations=True)
        else:
            h, intermediate_output = super().forward(obs, return_last_activations=True, 
                extra_fc_input=extra_fc_input,
                intermediate_output_layer=intermediate_output_layer)
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        if intermediate_output_layer == -1:
            return Independent(Normal(mean, std), 1)
        else:
            return Independent(Normal(mean, std), 1), intermediate_output


    def configure_optimizers(self):
        return self.optimizer, None


class GCConvPolicyWrapper:
    def __init__(self, policy):
        self.policy = policy
        self.no_goal = False

    def forward_np(self, image, goal_image, task_embedding=None):
        '''
        Image is (batch, height, width, channels)
        Data type is uint8 and normalized
        '''
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).to(ptu.device)
        if task_embedding is not None:
            task_embedding = torch.from_numpy(task_embedding)
       
        if not self.no_goal: 
            goal_image = goal_image.astype(np.float32) / 255.0
            goal_image = torch.from_numpy(goal_image).to(ptu.device)
            image_data = torch.stack([image, goal_image], dim=1)      #b, 2, h, w, c
            image_data = torch.einsum('b k h w c -> b k c h w', image_data)
            image_data = image_data.flatten(1, 2) # b k c h w -> b (k c) h w
        else:
            image_data = torch.einsum('b h w c -> b c h w', image)

        return self.policy(image_data, extra_fc_input=task_embedding)

    def get_action(self, image, goal_image, task_embedding=None):
        dist = self.forward_np(image, goal_image, task_embedding)
        action = dist.mean
        return action.detach().cpu().numpy()

    def compute_loss_np(self, image, goal_image, actions, goal_distances, task_embedding=None):
        dist = self.forward_np(image, goal_image, task_embedding)
        actions = torch.from_numpy(actions).to(ptu.device)

        policy_logpp = dist.log_prob(actions, )
        policy_loss = -policy_logpp.mean()

        action_pred = dist.mean 
        mse_loss = F.mse_loss(action_pred.float(), actions.float(), reduction="none").mean()

        return {'loss': policy_loss, 'mse_loss': mse_loss}


class RandomCrop:
    """
    Source: # https://github.com/pratogab/batch-transforms
    Applies the :class:`~torchvision.transforms.RandomCrop` transform to
    a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image.
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros((tensor.size(0), tensor.size(1),
                                  tensor.size(2) + self.padding * 2,
                                  tensor.size(3) + self.padding * 2),
                                 dtype=self.dtype, device=ptu.device)
            padded[:, :, self.padding:-self.padding,
            self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0, h - th + 1, (tensor.size(0),),
                              device=ptu.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=ptu.device)

        rows = torch.arange(th, dtype=torch.long, device=ptu.device) + i[:,
                                                                        None]
        columns = torch.arange(tw, dtype=torch.long, device=ptu.device) + j[:,
                                                                           None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:, torch.arange(tensor.size(0))[:, None, None],
                 rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)

class WarpPerspective:
    def __init__(self, size, warp_pixels=6, num_warps=1000):
        self.size = size
        self.warp_pixels = warp_pixels
        self.num_warps = num_warps
        src = np.array([[[0, 0], [0, size], [size, 0], [size, size]]])

        self.warps = []
        for i in range(num_warps):
            dst_jitter = np.random.uniform(-warp_pixels, warp_pixels, size=(1, 4, 2))
            dst = np.clip(src + dst_jitter, 0, self.size)
            self.warps.append(get_perspective_transform(ptu.from_numpy(src), ptu.from_numpy(dst)).detach().cpu().numpy()[0])
        self.warps = np.array(self.warps)

    def __call__(self, tensor):
        b, *_ = tensor.size()
        warp_matrix = ptu.from_numpy(self.warps[np.random.randint(0, self.num_warps, size=b)])
        return warp_perspective(tensor, warp_matrix, dsize=(self.size, self.size))

class WarpAffine:
    def __init__(self, size, warp_angle=10, num_warps=1000):
        self.size = size
        self.warp_angle = warp_angle
        self.num_warps = num_warps
        center = ptu.from_numpy(np.array([[size // 2, size // 2]]))
        scale = ptu.from_numpy(np.array([[1, 1]]))

        self.warps = []
        for i in range(num_warps):
            angle = ptu.from_numpy(np.random.uniform(-warp_angle, warp_angle, size=(1,)))
            self.warps.append(get_rotation_matrix2d(center, angle, scale).detach().cpu().numpy()[0])
        self.warps = np.array(self.warps)

    def __call__(self, tensor):
        b, *_ = tensor.size()
        warp_matrix = ptu.from_numpy(self.warps[np.random.randint(0, self.num_warps, size=b)])
        return warp_affine(tensor, warp_matrix, dsize=(self.size, self.size))
