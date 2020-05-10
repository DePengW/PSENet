#coding=utf-8
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import os
import math

cur_path = os.path.abspath(os.path.dirname(__file__))
ctw_root_dir = cur_path + '/data/CTW1500/'
ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
ctw_train_gt_dir = ctw_root_dir + 'train/text_label_curve/'
ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
ctw_test_gt_dir = ctw_root_dir + 'test/text_label_curve/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print img_path
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)
        
        bboxes.append(bbox)
        tags.append(True)
    return np.array(bboxes), tags



def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


def random_horizontal_flip_points(bboxes, img):  
    '''
    @msg: 图片水平镜像后，坐标点的映射位置
    @param {bboxes:反转前的坐标点[gt_bboxes, kernels_bboxes], img:图片} 
    @return: {bboxes:反转后的坐标点，flip_arg：是否需要反转}
    '''
    h, w = img.shape[0:2]
    flip_arg = random.random
    if flip_arg < 0.5:
        for idx in range(len(bboxes)):   
            for i in range(len(bboxes[idx])):
                bboxes[idx][i] =  np.array([mirror_position(h, w, pos[0], pos[1]) for pos in bboxes[idx][i]])
    return bboxes, flip_arg

def mirror_position(h,w,x,y):
    '''
    @msg: 水平镜像映射
    '''
    x_center,y_center = w/2,h/2
    y0 = y
    x0 = 2 * x_center - x
    return round(x0),round(y0)


def random_rotate_points(bboxes, img):
    '''
    @msg: 图片随机旋转后，坐标点的映射位置
    @param {bboxes:旋转前的坐标点, img:图片} 
    @return: {bboxes:旋转后的坐标点，angle：旋转角度}
    '''
    h, w = img.shape[0:2]
    x_center,y_center = w/2,h/2
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    bboxes_rotate = [] 
    for idx in range(len(bboxes)):
        for i in range(len(bboxes[idx])):
            bboxes[idx][i] = np.array([rotate_position(angle, pos[0], pos[1], x_center, y_center) for pos in bboxes[idx][i]])
        
        #解决越界问题
        clip = np.array(((0, 0), (0, h), (w, h), (w, 0))) 
        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPaths(bboxes[idx], pyclipper.PT_SUBJECT, True)   
        bbox_rotate = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD)
        bboxes_rotate.append(np.array(bbox_rotate))

    return bboxes_rotate, angle

def rotate_position(angle, x, y, x_center, y_center):
  '''
  @msg: 随机旋转
  '''  
  angle = math.radians(angle)
  x_rotate = (x - x_center)*math.cos(angle) + (y - y_center)*math.sin(angle) + x_center
  y_rotate = (y - y_center)*math.cos(angle) - (x - x_center)*math.sin(angle) + y_center
  return round(x_rotate), round(y_rotate)

def random_crop_points(img, img_size, bboxes, gt_text_temp):
    '''
    @msg:图片随机裁剪后，坐标点的映射位置
    '''
    h, w = img.shape[0:2]
    th, tw = img_size   
    if w == tw and h == th:
        return bboxes
    
    if random.random() > 3.0 / 8.0 and np.max(gt_text_temp) > 0: 
        tl = np.min(np.where(gt_text_temp > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(gt_text_temp > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        y_offset = random.randint(tl[0], br[0])
        x_offset = random.randint(tl[1], br[1])
    else:
        y_offset = random.randint(0, h - th)
        x_offset = random.randint(0, w - tw)
    
    bboxes_crop = []
    for idx in range(len(bboxes)):
        for i in range(len(bboxes[idx])): 
            bboxes[idx][i] = np.array([crop_position(pos[0], pos[1], x_offset, y_offset) for pos in bboxes[idx][i]])
        #解决越界问题
        clip = np.array(((0, 0), (0, th), (tw, th), (tw, 0)))
        pc = pyclipper.Pyclipper()
        pc.AddPath(clip, pyclipper.PT_CLIP, True)
        pc.AddPaths(bboxes[idx], pyclipper.PT_SUBJECT, True)
        bbox_crop = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD)
        bboxes_crop.append(np.array(bbox_crop))
        
    return bboxes_crop, x_offset, y_offset

def crop_position(x, y, x_offset, y_offset):
    '''
    @msg: 随机裁剪
    '''
    x = x - x_offset
    y = y- y_offset
    return round(x), round(y)

def random_horizontal_flip(img, flip_arg):
    '''
    @msg: 根据指定flip_arg，水平反转图片
    '''
    if flip_arg < 0.5:
        img = np.flip(img, axis=1).copy()
    return img

def random_rotate(img, angle):
    '''
    @msg: 根据指定angle，旋转图片
    '''
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation

def random_crop(img, img_size, x_offset, y_offset):
    '''
    @msg: 根据指定偏移量，裁剪图片
    '''
    h, w = img.shape[0:2]
    th, tw = img_size   
    if w == tw and h == th:
        return img

    i = y_offset
    j = x_offset
    
    img = img[i:i + th, j:j + tw, :]
    return img

def circle(img, img_path, bboxes): 
    '''
    @msg: 根据bboxes点的位置，在img图片上打印出来，依次来看rotate/flip/crop是否正确
    '''
    for i in range(bboxes.shape[0]):
        for j in range(len(bboxes[i])):
            cv2.circle(img, (bboxes[i][j][0], bboxes[i][j][1]), 2,(0, 255, 0), 3)
    save_path = cur_path + '/test_img/wdp_new_crop_label_{}'.format(img_path.split('/')[-1])
    cv2.imwrite(save_path, img)


def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

class CTW1500LoaderWdp(data.Dataset):
    def __init__(self, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale

        data_dirs = [ctw_train_data_dir]
        gt_dirs = [ctw_train_gt_dir]
        
        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                

                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
            

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)
        
        if self.is_transform:
            img = random_scale(img, self.img_size[0])

            #构建gt_text_temp(用于crop判断offset)、同时根据random_scale转换bboxes
            gt_text_temp = np.zeros(img.shape[0:2], dtype='uint8')   
            if bboxes.shape[0] > 0:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14), (bboxes.shape[0], bboxes.shape[1] / 2, 2)).astype('int32') #bboxes[0]:表示label有几个框，bboxes[1]:表示框的4个点，bboxes[2]：表示每个点的2维坐标
                for i in range(bboxes.shape[0]):
                    cv2.drawContours(gt_text_temp, [np.array(bboxes[i])], -1, i + 1, -1)

            #根据gt_bboxes构建kernels_bboxes
            kernels_bboxes = [] 
            for i in range(1, self.kernel_num):
                rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
                kernel_bboxes = shrink(bboxes, rate)   
                kernels_bboxes.append(kernel_bboxes)
            bboxes = [bboxes]
            bboxes.extend(kernels_bboxes)

            #将bboxes进行horizontal_flip（左右翻转）、rotate、crop操作
            bboxes_flip, flip_arg = random_horizontal_flip_points(bboxes, img)            
            bboxes_rotate, rotate_angle= random_rotate_points(bboxes_flip, img)
            bboxes_crop, x_offset, y_offset = random_crop_points(img, self.img_size, bboxes_rotate, gt_text_temp)
        
            gt_bboxes, kernels_bboxes = bboxes_crop[0], bboxes_crop[1:]

            #得到处理后的gt_text、gt_kernels
            gt_text = np.zeros(self.img_size, dtype='uint8')
            training_mask = np.ones(self.img_size, dtype='uint8')  
            for i in range(len(gt_bboxes)):
                cv2.drawContours(gt_text, [np.array(gt_bboxes[i])], -1, i + 1, -1)
                # if not tags[i]:
                #     cv2.drawContours(training_mask, [np.array(gt_bboxes[i])], -1, 0, -1)

            gt_kernels = []
            for i in range(1, self.kernel_num):
                gt_kernel = np.zeros(self.img_size, dtype='uint8')
                for j in range(kernels_bboxes[i-1].shape[0]):
                    cv2.drawContours(gt_kernel, [np.array(kernels_bboxes[i-1][j])], -1, 1, -1)
                gt_kernels.append(gt_kernel)
        else:
            gt_text = np.zeros(img.shape[0:2], dtype='uint8')
            training_mask = np.ones(img.shape[0:2], dtype='uint8')
            if bboxes.shape[0] > 0:
                bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14), (bboxes.shape[0], bboxes.shape[1] / 2, 2)).astype('int32')
                for i in range(bboxes.shape[0]):
                    cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                    if not tags[i]:
                        cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
            
            gt_kernals = []
            for i in range(1, self.kernel_num):
                rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
                gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
                kernal_bboxes = shrink(bboxes, rate)
                for i in range(bboxes.shape[0]):
                    cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
                gt_kernals.append(gt_kernal)

        #处理图片
        if self.is_transform:
            img = random_horizontal_flip(img, flip_arg)
            img = random_rotate(img, rotate_angle)       
            img = random_crop(img, self.img_size, x_offset, y_offset)
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')


        gt_text[gt_text > 0] = 1   
        gt_kernels = np.array(gt_kernels)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernels = torch.from_numpy(gt_kernels).float()
        training_mask = torch.from_numpy(training_mask).float()
        # '''

        return img, gt_text, gt_kernels, training_mask