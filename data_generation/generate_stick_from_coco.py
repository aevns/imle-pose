import sys
sys.path.append("..")

import json
import numpy as np
import copy

import torchvision.transforms as transforms
import cv2

root = 'data/coco/'
collection = 'train'
collection_json = 'annotations/person_keypoints_' + collection + '2017.json'

height = 64
width = 48
margin = 4
line_thickness = 2
dimensions = (height, width, 3)
colors = [
    (255, 255, 0),  # L calf
    (255, 128, 0),  # L thigh
    (255, 0, 0),    # R calf
    (255, 0, 128),  # R thigh
    (255, 0, 255),  # hips
    (128, 0, 255),  # L torso
    (0, 0, 255),    # R torso
    (0, 128, 255),  # shoulders
    (0, 255, 255),  # L humerous
    (0, 255, 128),  # R humerous
    (0, 255, 0),    # L forearm
    (128, 255, 0),  # R forearm
    (255, 128, 128),# forehead
    (128, 255, 128),# L face
    (128, 128, 255),# R face
    (255, 255, 128),# L head
    (255, 128, 255),# R head
    (128, 255, 255),# L neck
    (255, 255, 255) # R neck
]

convert_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((height, width))
])

with open(root + collection_json, 'r') as fid:
    file = json.load(fid)

file_out = {}
file_out['info'] = copy.deepcopy(file['info'])
file_out['info']['description'] = 'Stick Dataset (generated from COCO 2017 Dataset)'
file_out['licenses'] = copy.deepcopy(file['licenses'])
file_out['images'] = {}
file_out['annotations'] = {}
file_out['categories'] = copy.deepcopy(file['categories'])

skeleton = file['categories'][0]['skeleton']

for i in range(0, len(file['annotations'])):

    if file['annotations'][i]['iscrowd'] == 0:
        
        # get the associated image
        image_id = file['annotations'][i]['image_id']
        img = next((x for x, y in enumerate(file['images']) if y['id'] == image_id), -1)
        w = file['images'][img]['width']
        h = file['images'][img]['height']

        # get the keypoints; if any important labels are missing, move on to the next annotation
        keypoints = file['annotations'][i]['keypoints']
        if any(k == 0 for k in keypoints[2::3]):
            continue

        # determine the cropped region
        minx = min([keypoints[::3][x] for x in range(0, len(keypoints[::3])) if keypoints[2::3][x] > 0])
        maxx = max([keypoints[::3][x] for x in range(0, len(keypoints[::3])) if keypoints[2::3][x] > 0])
        miny = min([keypoints[1::3][x] for x in range(0, len(keypoints[1::3])) if keypoints[2::3][x] > 0])
        maxy = max([keypoints[1::3][x] for x in range(0, len(keypoints[1::3])) if keypoints[2::3][x] > 0])
        crop_center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
        crop_scale = max((maxx - minx + 2 * margin)/width, (maxy - miny + 2 * margin)/height)
        if crop_scale <= 0:
            continue
        
        # write new annotation data
        annotation = copy.deepcopy(file['annotations'][i])
        for k in range(0, len(keypoints[::3])):
            annotation['keypoints'][3*k] = ((keypoints[3*k] - crop_center[0]) / crop_scale + width / 2).astype(np.uint8).item()
            annotation['keypoints'][3*k+1] = ((keypoints[3*k+1] - crop_center[1]) / crop_scale + height / 2).astype(np.uint8).item()
        annotation['segmentation'] = None
        annotation['area'] = None
        annotation['bbox'] = [0, 0, height, width]
        annotation['image_id'] = annotation['id']
        file_out['annotations'][i] = annotation

        # write new image info
        image_data = copy.deepcopy(file['images'][img])
        image_data['file_name'] = str(annotation['image_id']) + '.png'
        image_data['id'] = annotation['image_id']
        image_data['height'] = height
        image_data['width'] = width
        file_out['images'][i] = image_data
        
        # generate new image
        bmp = np.zeros(dimensions, np.uint)
        num_bones = 0
        for b in range(0, len(skeleton)):
            # skeleton bones are indexed from 1 in the annotation data
            child = skeleton[b][0] - 1
            parent = skeleton[b][1] - 1

            # get bone positions
            start = annotation['keypoints'][3*child:3*child+2]
            end = annotation['keypoints'][3*parent:3*parent+2]

            # add to image
            tmp = np.zeros(dimensions, np.uint8)
            cv2.line(tmp, tuple(start), tuple(end), colors[b], line_thickness)
            bmp += tmp//4
            bmp = np.minimum(bmp, 255)
        
        bmp = bmp.astype(np.uint8)
        cv2.imwrite('data/stick/' + collection + '/' + image_data['file_name'], bmp)

# save new annotation data
with open('data/stick/annotations/person_keypoints_' + collection + '.json', 'w+') as f:
    json.dump(file_out, f)