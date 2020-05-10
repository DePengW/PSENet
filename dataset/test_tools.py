#coding=utf-8
import os
from PIL import Image
import cv2

#-------------------drawContours函数-----------
# import cv2
# import numpy as np
# img=cv2.imread(img_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 第三步：对图片做二值变化
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # 第四步：获得图片的轮廓值
# contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# # 第五步：在图片中画出图片的轮廓值
# draw_img = img.copy()
# ret = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# # 第六步：画出带有轮廓的原始图片
# cv2.imwrite(save_path, ret)



#-------------------越界处理函数-----------
import pyclipper
import numpy as np

subj = (
    ((180, 200), (260, 200), (260, 150), (180, 150)),
    ((215, 160), (230, 190), (200, 190))
)


subj = np.array(subj)

a = np.expand_dims(subj,0).repeat(2,axis=0)


clip = ((190, 210), (240, 210), (240, 130), (190, 130))

clip = np.array(clip)

pc = pyclipper.Pyclipper()

pc.AddPath(clip, pyclipper.PT_CLIP, True)
pc.AddPaths(a, pyclipper.PT_SUBJECT, True)

solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD)
print(solution)


#-------------------打印mix图（mask和原图融合）-----------
# import numpy as np
# imgfile = '/data/home/depwang/code/PSENet/dataset/test_img/new_img_0451.jpg'
# pngfile = '/data/home/depwang/code/PSENet/dataset/test_img/new_mask_0451.jpg'
# img = cv2.imread(imgfile, 1)
# mask = cv2.imread(pngfile, 0)

# contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

# img = img[:, :, ::-1]
# img[..., 2] = np.where(mask == 1, 255, img[..., 2])

# save_path = '/data/home/depwang/code/PSENet/dataset/test_img/mix_0451.jpg'   #yuan
# # save_path = '/data/home/depwang/code/PSENet/dataset/test_img/wdp_mix_0548.jpg'
# cv2.imwrite(save_path, img)


#-------------------测试对比dataloader速度-----------
# import torch
# import os
# import sys
# import time
# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)
# from dataset import IC15Loader
# from dataset import IC15LoaderWdp
# from dataset import CTW1500Loader
# from dataset import CTW1500LoaderWdp

# def speedtest(batch_size):
#     #add wdp for keypoints
#     # data_loader = IC15LoaderWdp(is_transform=True, img_size=640, kernel_num=7, min_scale=0.4)

#     data_loader = CTW1500LoaderWdp(is_transform=True, img_size=640, kernel_num=7, min_scale=0.4)
#     train_loader = torch.utils.data.DataLoader(
#         data_loader,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         drop_last=True,
#         pin_memory=True)
 
#     for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
        
#         #warmup 
#         if batch_idx == 2 :
#             start = time.time()
#             continue
#         if batch_idx == 5:
#             print("Speed per batch_size: {} /s".format( (time.time() - start) / 3) )
#             break

#     print("over")


# speedtest(batch_size = 32)





