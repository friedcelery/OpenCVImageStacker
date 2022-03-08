#!/usr/bin/env python3

import sys
import numpy as np
import cv2 as cv
import os

from tqdm import tqdm

def enumerateImages(images):
    file_list = os.listdir()

    if 'output' not in file_list:
        os.mkdir('output')

    image_list = []

    for i in file_list:
        if i.lower().endswith(('.png','.jpg','.bmp')):
            image_list.append(i)

    print("Detected images: ")
    for i in image_list:
        print(i)

    images = []
    for i in image_list:
        images.append(cv.imread(i))

    return images


def detectFeaturesORB(images):
    print("Detecting features")
    orb = cv.ORB_create()

    kp_list = []
    desc_list = []

    for i in tqdm(images):
        kp_temp, desc_temp = orb.detectAndCompute(i,None)
        kp_list.append(kp_temp)
        desc_list.append(desc_temp)
    return kp_list,desc_list


def warpAndStack(images,kp_list,desc_list):
    final_image = images[0].astype(np.float32)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    print("Aligning and Stacking images")
    for i in tqdm(range(1,len(images))):
        matches = bf.match(desc_list[0],desc_list[i])
        matches = sorted(matches, key = lambda x:x.distance)

        src_pts = np.float32(
            [kp_list[0][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32(
            [kp_list[i][m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        M,_ = cv.findHomography(dst_pts,src_pts, cv.RANSAC, 5)
        w,h,_ = images[i].shape
        
        warp_img = cv.warpPerspective(images[i].astype(np.float32),M,(h,w))
        
        final_image += warp_img

    final_image = final_image/len(images)

    final_image = final_image.astype(np.uint8)
    return final_image



if __name__ == '__main__':
    images = []
    images = enumerateImages(images)
    kp_list , desc_list = detectFeaturesORB(images)
    final_image = warpAndStack(images,kp_list,desc_list)

    write_path = os.path.join(os.getcwd(),'output/output.jpg')
    print("Output saved to",write_path)
    cv.imwrite(os.path.join(os.getcwd(),'output/output.jpg'),final_image)

