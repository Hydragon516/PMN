import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from fast_slic import Slic
import cv2
import config

#several data augumentation strategies
def cv_random_flip(img, label, flow):
    flip_flag = random.randint(0, 1)

    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        flow = flow.transpose(Image.FLIP_LEFT_RIGHT)

    return img, label, flow

def randomCrop(image, label, flow):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    return image.crop(random_region), label.crop(random_region), flow.crop(random_region)

def randomRotation(image,label,flow):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        flow = flow.rotate(random_angle, mode)
    
    return image,label,flow

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
    def __init__(self, image_root, gt_root, flow_root, trainsize):
        self.trainsize = trainsize
        self.images = []
        self.gts = []
        self.flows = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg') or f.endswith('.png')]

        self._class = {}

        cnt = 0
        for i in range(len(self.flows)):
            if (self.flows[i].split("/")[-1]).split("_")[0] not in self._class:
                self._class[(self.flows[i].split("/")[-1]).split("_")[0]] = cnt
                cnt += 1
            
            self.images.append(image_root + self.flows[i].split("/")[-1])
            self.gts.append(gt_root + (self.flows[i].split("/")[-1]).replace(".jpg", ".png"))
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.flows = sorted(self.flows)
        
        self.size = len(self.images)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.flows_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        _class = self._class[(self.images[index].split("/")[-1]).split("_")[0]]
        gt = self.binary_loader(self.gts[index])
        flow = self.rgb_loader(self.flows[index])
        
        image, gt, flow = cv_random_flip(image, gt, flow)
        image, gt, flow = randomCrop(image, gt, flow)
        image, gt, flow = randomRotation(image, gt, flow)
        
        image = colorEnhance(image)

        np_img = np.array(image)
        np_img = cv2.resize(np_img, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        
        np_gt = np.array(gt)
        np_gt = cv2.resize(np_gt, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR) / 255

        np_flow = np.array(flow)
        np_flow = cv2.resize(np_flow, dsize=(self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)

        slic = Slic(num_components=config.TRAIN['num_components'], compactness=10)
        SS_map = slic.iterate(np_img)
        flow_SS_map = slic.iterate(np_flow)
        
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

        ###
        flow_SS_map = flow_SS_map + 1

        flow_SS_maps = []
        flow_SS_maps_label = []

        for i in range(1, config.TRAIN['num_components'] + 1):
            buffer = np.copy(flow_SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    flow_SS_maps_label.append(1)
                else:
                    flow_SS_maps_label.append(0)
            else:
                flow_SS_maps_label.append(0)

            flow_SS_maps.append(buffer)
        
        flow_ss_map = np.array(flow_SS_maps)
        flow_ss_maps_label = np.array(flow_SS_maps_label)

        ###
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        flow = self.flows_transform(flow)
        
        return image, gt, flow, ss_map, ss_maps_label, flow_ss_map, flow_ss_maps_label, _class

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')

    def resize(self, img, gt, flow):
        assert img.size == gt.size and gt.size == flow.size
        
        w, h = img.size
        
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST),flow.resize((w, h), Image.NEAREST)
        else:
            return img, gt, flow

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, flow_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset(image_root, gt_root, flow_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader
    
class SalObjDataset_test(data.Dataset):
    def __init__(self, val_image_root, valid_list, testsize):
        self.testsize = testsize
        
        self.images = []
        self.gts = []
        self.flows = []
        
        for valid_name in valid_list:
            image_root = os.path.join(val_image_root, valid_name, "RGB") + "/"
            gt_root = os.path.join(val_image_root, valid_name, "GT") + "/"
            flow_root = os.path.join(val_image_root, valid_name, "depth") + "/"

            new_images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
            new_gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            new_flows = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg') or f.endswith('.png')]

            new_images = sorted(new_images)
            new_gts = sorted(new_gts)
            new_flows = sorted(new_flows)

            for i in range(len(new_flows)):
                self.images.append(new_images[i])
                self.gts.append(new_gts[i])
                self.flows.append(new_flows[i])
        
        self._class = {}

        cnt = 0
        for i in range(len(self.images)):
            if (self.images[i].split("/")[-1]).split("_")[0] not in self._class:
                self._class[(self.images[i].split("/")[-1]).split("_")[0]] = cnt
                cnt += 1

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.flows = sorted(self.flows)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.flows_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)), 
            transforms.ToTensor()])
        
        self.size = len(self.images)
    
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        _class = self._class[(self.images[index].split("/")[-1]).split("_")[0]]
        gt = self.binary_loader(self.gts[index])

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
        flow = self.rgb_loader(self.flows[index])

        np_flow = np.array(flow)
        np_flow = cv2.resize(np_flow, dsize=(self.testsize, self.testsize), interpolation=cv2.INTER_LINEAR)

        flow_SS_map = slic.iterate(np_flow)

        flow_SS_map = flow_SS_map + 1

        flow_SS_maps = []
        flow_SS_maps_label = []

        for i in range(1, config.TRAIN['num_components'] + 1):
            buffer = np.copy(flow_SS_map)
            buffer[buffer != i] = 0
            buffer[buffer == i] = 1

            if np.sum(buffer) != 0:
                if (np.sum(buffer * np_gt) / np.sum(buffer)) > 0.5:
                    flow_SS_maps_label.append(1)
                else:
                    flow_SS_maps_label.append(0)
            else:
                flow_SS_maps_label.append(0)

            flow_SS_maps.append(buffer)
        
        flow_ss_map = np.array(flow_SS_maps)
        flow_ss_maps_label = np.array(flow_SS_maps_label)

        flow = self.flows_transform(flow)
        
        name = self.images[index].split('/')[-1]
        valid_name = self.images[index].split('/')[-3]
        
        image_for_post = self.rgb_loader(self.images[index])
        image_for_post = image_for_post.resize((self.testsize, self.testsize))
        
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        info = [gt.size, valid_name, name]
        
        gt = self.gt_transform(gt)
        
        return image, gt, flow, info, np.array(image_for_post), ss_map, ss_maps_label, flow_ss_map, flow_ss_maps_label, _class

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            
            return img.convert('L')
    
    def __len__(self):
        return self.size

def get_testloader(val_image_root, valid_list, batchsize, testsize, shuffle=False, num_workers=12, pin_memory=False):

    dataset = SalObjDataset_test(val_image_root, valid_list, testsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader