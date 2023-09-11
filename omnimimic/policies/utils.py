import torch
from torchvision.transforms import ColorJitter

def get_augmentation_transforms(augmentation_types, augmentation_config):
    input_dim = augmentation_config['input_dim'] #Assume square image for now
    transforms = []

    for augmentation_type in augmentation_types:
        if augmentation_type == 'random_crop':
            #padding = augmentation_config['random_crop_padding']
            transforms.append(RandomCrop(
                input_dim, input_dim//16, device='cuda'))
        elif augmentation_type == 'warp_perspective':
            transforms.append(WarpPerspective(
                input_dim))
        elif augmentation_type == 'warp_affine':
            transforms.append(WarpAffine(
                input_dim))
        elif augmentation_type == 'color_jitter':
            transforms.append(ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(1),
                saturation=(0.5, 1.5),
                hue=(-0.1, 0.1)
            ))
    return transforms

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
                                 dtype=self.dtype, device=self.device)
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
                              device=self.device)
            j = torch.randint(0, w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,
                                                                        None]
        columns = torch.arange(tw, dtype=torch.long, device=self.device) + j[:,
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

