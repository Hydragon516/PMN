import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from fast_slic import Slic
import cv2
import config

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        
        return img.convert('L')


def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5,15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0,20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0,30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        
        return im
    
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])

    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0]-1)  
        randY = random.randint(0, img.shape[1]-1)  

        if random.randint(0,1) == 0:  
            img[randX, randY] = 0  
        else:  
            img[randX, randY] = 255 

    return Image.fromarray(img)  


class SalObjDataset(data.Dataset):
    def __init__(self):
        self.trainsize = config.TRAIN['img_size']
        image_root = os.path.join(config.DATA['data_root'], config.DATA['pretrain'], "RGB") + "/"
        gt_root = os.path.join(config.DATA['data_root'], config.DATA['pretrain'], "GT") + "/"
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.filter_files()
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        
        images = []
        gts = []
        
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        
        self.images = images
        self.gts = gts
    
    def __getitem__(self, index):
        image = rgb_loader(self.images[index])
        gt = binary_loader(self.gts[index])
        
        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        
        image = colorEnhance(image)

        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        
        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR) / 255

        slic = Slic(num_components=config.TRAIN['num_components'], compactness=10)
        SS_map = slic.iterate(np_img)
        
        ###
        SS_map = SS_map + 1
        
        SS_maps = []
        SS_maps_label = []

        for i in range(1, config.TRAIN['num_components'] + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    SS_maps_label.append(1)
                else:
                    SS_maps_label.append(0)
            else:
                SS_maps_label.append(0)

            SS_maps.append(buffer)
        
        ss_map = np.array(SS_maps)
        ss_maps_label = np.array(SS_maps_label)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt, ss_map, ss_maps_label

    def __len__(self):
        return self.size

    
class SalObjDataset_test(data.Dataset):
    def __init__(self):
        self.testsize = config.TRAIN['img_size']
        self.images = []
        self.gts = []

        folder = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_val'])
        valid_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
        
        for valid_name in valid_list:
            image_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_val'], valid_name, "RGB") + "/"
            gt_root = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_val'], valid_name, "GT") + "/"

            new_images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            new_gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

            for i in range(len(new_images)):
                self.images.append(new_images[i])
                self.gts.append(new_gts[i])
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

        self.size = len(self.images)
    
    def __getitem__(self, index):
        image = rgb_loader(self.images[index])
        gt = binary_loader(self.gts[index])

        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)
        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR) / 255
 
        slic = Slic(num_components=config.TRAIN['num_components'], compactness=10)
        SS_map = slic.iterate(np_img)
        
        SS_map = SS_map + 1

        SS_maps = []
        SS_maps_label = []

        for i in range(1, config.TRAIN['num_components'] + 1):
            buffer = np.copy(SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    SS_maps_label.append(1)
                else:
                    SS_maps_label.append(0)
            else:
                SS_maps_label.append(0)

            SS_maps.append(buffer)
        
        ss_map = np.array(SS_maps)
        ss_maps_label = np.array(SS_maps_label)

        image = self.transform(image)
        
        name = self.images[index].split('/')[-1]
        valid_name = self.images[index].split('/')[-3]
        
        image_for_post = rgb_loader(self.images[index])
        image_for_post = image_for_post.resize((self.testsize, self.testsize))
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        info = [gt.size, valid_name, name]
        
        gt = self.gt_transform(gt)
        
        return image, gt, info, np.array(image_for_post), ss_map, ss_maps_label
    
    def __len__(self):
        return self.size


def get_loader(shuffle=True, num_workers=12, pin_memory=False):
    dataset = SalObjDataset()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.TRAIN['batch_size'],
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    
    return data_loader


def get_testloader(shuffle=False, num_workers=12, pin_memory=False):
    dataset = SalObjDataset_test()
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.TRAIN['batch_size'],
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader