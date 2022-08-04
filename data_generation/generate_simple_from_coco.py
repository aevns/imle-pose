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

height = 128
width = 96
margin = 12
output_size = np.array([width, height])

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
        image_source = root + collection + '2017/' + file['images'][img]['file_name']
        bmp = cv2.imread(image_source)
        p0 = crop_center + crop_scale / 2 * output_size * np.array([-1,-1])
        p1 = crop_center + crop_scale / 2 * output_size * np.array([1,-1])
        p2 = crop_center + crop_scale / 2 * output_size * np.array([-1,1])
        t = cv2.getAffineTransform(np.float32([p0, p1, p2]), np.float32([[0,0],[width,0],[0,height]]))
        bmp = cv2.warpAffine(bmp, t, (width, height), flags=cv2.INTER_LINEAR)
        cv2.imwrite('data/simple/' + collection + '/' + image_data['file_name'], bmp)

# save new annotation data
with open('data/simple/annotations/person_keypoints_' + collection + '.json', 'w+') as f:
    json.dump(file_out, f)