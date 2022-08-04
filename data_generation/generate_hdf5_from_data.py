import sys
sys.path.append("..")

import torch
import torchvision.transforms as transforms
import json
import cv2
import h5py

root = 'data/simple/'
collections = ['train','val']

for collection in collections:
    collection_json = 'annotations/person_keypoints_' + collection + '.json'

    with open(root + collection_json, 'r') as fid:
        file = json.load(fid)

    count = len(file['annotations'])

    example = next(iter(file['annotations']))
    bones = len(file['annotations'][example]['keypoints'])//3

    example = next(iter(file['images']))
    w = file['images'][example]['width']
    h = file['images'][example]['height']

    images = torch.empty((count, 3, h, w))
    poses = torch.empty((count, bones, 2))

    convert_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    i = 0
    for k in file['annotations']:
        kp = file['annotations'][k]['keypoints']
        id = file['annotations'][k]['image_id']
        
        for j in range(0, bones):
            poses[i,j,0] = kp[3*j]
            poses[i,j,1] = kp[3*j+1]

        img = cv2.imread(root + collection + '/' + file['images'][k]['file_name'])

        images[i] = convert_tensor(img)
        i += 1

    if collection == 'train':
        mean_color = torch.mean(images,dim=(0,2,3))
        std_color = torch.std(images,dim=(0,2,3))

    #generate hdf5 file
    write_file = h5py.File(root + collection + '.hdf5', 'w')
    write_file['images'] = images
    write_file['poses'] = poses
    write_file['keypoints'] = file['categories'][0]['keypoints']
    write_file['skeleton'] = file['categories'][0]['skeleton']
    write_file['mean_color'] = mean_color
    write_file['std_color'] = std_color
    write_file.close()