import sys
sys.path.append("..")

import torch
import torchvision.transforms as transforms
import json
import cv2
import h5py

root = 'quick_data/complete/'
collections = ['train','val']
max_chunk_size = 1000

for collection in collections:

    # reading annotation data
    collection_json = 'annotations/person_keypoints_' + collection + '.json'

    with open(root + collection_json, 'r') as fid:
        file = json.load(fid)

    count = len(file['annotations'])

    example = next(iter(file['annotations']))
    bones = len(file['annotations'][example]['keypoints'])//3

    example = next(iter(file['images']))
    w = file['images'][example]['width']
    h = file['images'][example]['height']

    convert_tensor = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # writing annotation data to hdf5
    with h5py.File(root + collection + '.hdf5', 'w') as write_file:
        count = len(file['annotations'])
        chunks = 1 + (count - 1)//max_chunk_size

        write_file['keypoints'] = file['categories'][0]['keypoints']
        write_file['skeleton'] = file['categories'][0]['skeleton']
        image_set = write_file.create_dataset('images', (0, 3, h, w), maxshape=(count, 3, h, w), chunks=True)
        pose_set = write_file.create_dataset('poses', (0, bones, 3), maxshape=(count, bones, 3), chunks=True)

        if collection == 'train':
            mean_color = torch.zeros((3,))
            std_color = torch.zeros((3,))

        total_count = 0
        prev_count = 0
        chunk = 0
        i = 0

        chunk_size = min(max_chunk_size, count - chunk * max_chunk_size)
        images = torch.empty((max_chunk_size, 3, h, w))
        poses = torch.empty((max_chunk_size, bones, 3))

        for k in file['annotations']:
            if (i == chunk_size):
                chunk_size = min(max_chunk_size, count - chunk * max_chunk_size)
                images = torch.empty((chunk_size, 3, h, w))
                poses = torch.empty((chunk_size, bones, 3))
                chunk += 1
                i = 0
            
            kp = file['annotations'][k]['keypoints']
            id = file['annotations'][k]['image_id']
            
            for j in range(0, bones):
                poses[i,j,0] = kp[3*j]
                poses[i,j,1] = kp[3*j+1]
                poses[i,j,2] = kp[3*j+2]

            img = cv2.imread(root + collection + '/' + file['images'][k]['file_name'])

            images[i] = convert_tensor(img)

            total_count += 1
            i += 1
            
            if (i == chunk_size):
                if collection == 'train':
                    mean_color += torch.mean(images, dim=(0,2,3)) / chunks
                    std_color += torch.std(images, dim=(0,2,3)) / chunks

                image_set.resize(total_count, axis=0)
                pose_set.resize(total_count, axis=0)
                image_set[prev_count:] = images
                pose_set[prev_count:] = poses
                prev_count = total_count
        
        write_file['mean_color'] = mean_color
        write_file['std_color'] = std_color
        write_file.close()