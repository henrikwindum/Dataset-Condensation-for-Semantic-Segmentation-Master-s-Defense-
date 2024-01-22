import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate

from torchvision.utils import make_grid

from networks import UNet, ConvNet

class Config:
    # ['American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Boxer', 'Chihuahua', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger', 'Miniature Pinscher', 'Newfoundland', 'Pomeranian', 'Pug', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']
    oxford_dogs = {1, 2, 3, 4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 25, 28, 29, 30, 31, 34, 35, 36}

    # subset of dog races that doesn't resemble cats
    oxford_dogs_subset = [1, 2, 3, 4, 8, 12, 13, 14, 15, 16, 18, 19, 21, 22]

    # ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Egyptian Mau', 'Maine Coon', 'Persian', 'Ragdoll', 'Russian Blue', 'Siamese', 'Sphynx']
    oxford_cats = {0, 5, 6, 7, 9, 11, 20, 23, 26, 27, 32, 33}

    dict = {
        "oxford_dogs" : oxford_dogs,
        "oxford_cats" : oxford_cats,
    }

config = Config()


def preprocess_mask(mask):
    mask = torch.as_tensor(np.array(mask, copy=True), dtype=torch.float)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


class CostumTargetTransform:
    def __init__(self, num_classes, oxford_cats, oxford_dogs, im_size=(128,128)):
        self.num_classes = num_classes
        self.oxford_cats = oxford_cats
        self.oxford_dogs = oxford_dogs
        self.mask_transform = transforms.Compose([transforms.Resize(im_size, interpolation=transforms.InterpolationMode.NEAREST), transforms.CenterCrop(im_size)])

    def __call__(self, target):
        label, mask = target
        mask = preprocess_mask(self.mask_transform(mask))
        if self.num_classes == 2:
            label = 0
            class_probabilities = torch.stack([1 - mask, mask], axis=0)
        elif self.num_classes == 3:
            # For multi-class classification (background, cats, dogs)
            class_probabilities = torch.zeros((self.num_classes, *mask.shape), dtype=torch.float)

            # Background class probabolity
            class_probabilities[0,:,:] = (mask == 0).float()

            if label in self.oxford_cats:
                # Cat class probability
                class_probabilities[1,:,:] = (mask == 1).float()
                label = 0
            elif label in self.oxford_dogs:
                # Dog class probability
                class_probabilities[2,:,:] = (mask == 1).float()
                label = 1
            else:
                raise ValueError("Label not in cat or dog labels")
        else:
            raise ValueError("num_classes must be 2 or 3")
        return (label, class_probabilities)


def get_dataset(dataset, data_path, args=None):
    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == "Oxford-IIIT":
        channel = 3
        im_size = (128, 128)
        if args.seg_task == "B":
            num_classes = 2
        elif args.seg_task == "M":
            num_classes = 3
        else:
            exit("DC error: unknown segmentation task")
        mean = [0.4783, 0.4459, 0.3957]
        std = [0.2652, 0.2598, 0.2679]

        # if args.zca:
        #     im_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size), transforms.CenterCrop(im_size)])
        # else:
        #     im_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize(im_size), transforms.CenterCrop(im_size)])

        im_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Resize(im_size), transforms.CenterCrop(im_size)])
        target_transform = CostumTargetTransform(num_classes, config.oxford_cats, config.oxford_dogs)

        dst_train = datasets.OxfordIIITPet(data_path, split="trainval", download=True, transform=im_transform, target_transform=target_transform, target_types=("category", "segmentation")) # no augmentation
        dst_test = datasets.OxfordIIITPet(data_path, split="test", download=True, transform=im_transform, target_transform=target_transform, target_types=("category", "segmentation"))

    else:
        exit("unknown dataset: %s"%dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_real, shuffle=False, num_workers=2)

    return channel, im_size, num_classes, mean, std, dst_train, dst_test, testloader


def get_network(model, channel, num_classes, im_size=(128, 128)):
    if model == 'UNetV1':
        net = UNet(in_channels=channel, out_channels=num_classes, channels=[8, 16, 32, 64])
    elif model == 'UNetV2':
        net = UNet(in_channels=channel, out_channels=num_classes, channels=[12, 24, 48, 96])
    else:
        net = None
        exit('DC error: unknown model')
    return net


class TensorDataset(Dataset):
    def __init__(self, images, masks): # images: n x c x h x w tensor, masks: n x 1 x h x w tensor
        self.images = images.detach().float()
        self.masks = masks.detach().float()

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return self.images.shape[0]

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if mode == "train":
        net.train()
    else:
        net.eval()

    for i, datum in enumerate(dataloader, start=1):
        try:
            images, (_, target) = datum
        except:
            images, target = datum

        images = images.to(args.device)
        target = target.to(args.device)

        if aug:
            if args.dsa:
                images, target = DiffAugment(images, target, args.dsa_strategy, param=args.dsa_param)
            else:
                images, target = augment(images, target, args.dc_aug_param, device=args.device)

        n_b = images.shape[0]

        output = net(images)
        loss = criterion(output, target)

        loss_avg += loss.item() * n_b
        acc_avg += dice_coefficient_multilabel(output.detach(), target).item()
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= i

    return loss_avg, acc_avg

def evaluate_synset(it_eval, net, images_train, target_train, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    target_train = target_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, target_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()

    acc_train_list = []
    loss_train_list = []
    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)

        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start

    print("%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f" % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test

def dice_coefficient(output, target, eps=1e-7):
    output_flat = output.reshape(-1)
    target_flat = target.reshape(-1)

    intersection = (output_flat * target_flat).sum()

    return (2. * intersection + eps) / (output_flat.sum() + target_flat.sum() + eps)

def dice_coefficient_multilabel(output, target):
    output_prob = output.softmax(dim=1)

    dice = 0
    for c in range(target.shape[1]):
        dice += dice_coefficient(output_prob[:,c,:,:], target[:,c,:,:])
    return dice / target.shape[1]


def continuous_dice_coefficient(output, target):
    output_prob = torch.softmax(output, dim=1)

    output_flat = output_prob.view(output_prob.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersect = torch.sum(output_flat * target_flat, dim=1)
    output_size = torch.sum(output_flat, dim=1)
    target_size = torch.sum(target_flat, dim=1)

    c = torch.where(
        intersect > 0,
        torch.sum(output_flat * target_flat, dim=1) / torch.sum(output_flat * torch.sign(target_flat), dim=1),
        torch.tensor(1.0).to(output.device)
    )

    cDC = (2 * intersect) / (c * output_size + target_size)
    return torch.mean(cDC)


def plot_image_mask_pairs(images, masks, mean, std, args):
    masks = masks[:, 1, :, :].unsqueeze(1)

    images = images * std.view(3, 1, 1).to(images.device) + mean.view(3, 1, 1).to(images.device)

    images_grid = make_grid(images, nrow=10)
    masks_grid = make_grid(masks, nrow=10)

    if args.ipc == 10:
        figsize=(40, 5)
    else:
        figsize=(40, 40)

    # Plotting the grids
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    # Plot cat images and masks
    axs[0].imshow(images_grid.permute(1, 2, 0))
    axs[0].axis('off')

    # Plot dog images and masks
    axs[1].imshow(masks_grid.permute(1, 2, 0).squeeze(-1), cmap='gray')
    axs[1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

def get_daparam(dataset, model, model_eval, ipc):
    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.02
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'
    return dc_aug_param


def augment(images, y, dc_aug_param, device):
    if dc_aug_param is not None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1], shape[2]+crop*2, shape[3]+crop*2, dtype=torch.float, device=device)
            tg_ = torch.zeros(y.shape[1], shape[2]+crop*2, shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            tg_[:, crop:crop+shape[2], crop:crop+shape[3]] = y[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]
            y[i] = tg_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[3])
            tmp_img = F.interpolate(images[i:i+1], [h, w], mode='bilinear', align_corners=False)[0]
            tmp_y = F.interpolate(y[i:i+1], [h, w], mode='nearest')[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            tg_ = torch.zeros(y.shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp_img
            tg_[:, r:r + h, c:c + w] = tmp_y
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]
            y[i] = tg_[:, r:r + shape[2], c:c + shape[3]]


        def rotatefun(i):
            angle = np.random.randint(-rotate, rotate)
            im_ = scipyrotate(images[i].cpu().numpy(), angle=angle, axes=(-2, -1), reshape=False, mode='constant', cval=np.mean(mean))
            tg_ = scipyrotate(y[i].cpu().numpy(), angle=angle, axes=(-2, -1), reshape=False, mode='constant', cval=0, prefilter=False)
            images[i] = torch.tensor(im_, dtype=torch.float, device=device)
            y[i] = torch.tensor(tg_, dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.choice(augs)  # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images, y


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def DiffAugment(x, y, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x, y

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x, y = f(x, y, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
        y = y.contiguous()
    return x, y


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, y, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    y = F.grid_sample(y, grid, align_corners=True)
    return x, y


def rand_rotate(x, y, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    y = F.grid_sample(y, grid, align_corners=True)
    return x, y


def rand_flip(x, y, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    x_flipped = torch.where(randf < prob, x.flip(3), x)
    y_flipped = torch.where(randf < prob, y.flip(3), y)
    return x_flipped, y_flipped


def rand_brightness(x, y, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x, y


def rand_saturation(x, y, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x, y


def rand_contrast(x, y, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x, y


def rand_crop(x, y, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    y_pad = F.pad(y, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    y = y_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x, y


def rand_cutout(x, y, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    y = y * mask.unsqueeze(1)
    return x, y


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}