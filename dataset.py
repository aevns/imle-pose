import numpy as np
import torch
import torchvision.transforms as transforms
import h5py

FloatTensor =  torch.cuda.FloatTensor
Device = "cuda:0"

class StickDataset(torch.utils.data.Dataset):
    def __init__(self, heatmap_size, data_file = "./data/stick/train.hdf5"):
        super(StickDataset).__init__();

        with h5py.File(data_file, 'r') as df:
            self.poses = torch.from_numpy(df['poses'][...]).to(Device)
            self.images  = torch.from_numpy(df['images'][...]).to(Device)
            self.keypoints = df['keypoints'][...]
            self.skeleton = df['skeleton'][...]
        
        self._heatmap_size = heatmap_size
        self._sigma = 2
        self._feat_stride = np.array(self.images[0,0].shape) / np.array(heatmap_size)

        # Dataset Specific (stick)
        self.mean = FloatTensor([0.0280, 0.0251, 0.0286])
        self.std = FloatTensor([0.0927, 0.0904, 0.1053])

        # Generic (alphapose)
        #self.mean = FloatTensor([0.406, 0.457, 0.480])
        #self.std = FloatTensor([1., 1., 1.])

        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = 1/self.std),
                                               transforms.Normalize(mean = -self.mean, std = [ 1., 1., 1. ])])
    
    # converted from SimplePose target generator
    # joints_3d -> joints_2d
    def _target_generator(self, joints_2d):
        num_joints = len(self.keypoints)
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]),
                            dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_2d[i, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_2d[i, 1] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return torch.tensor(target, device="cuda:0"), np.expand_dims(target_weight, -1)
    
    def __len__(self):
        return self.poses.shape[0]
    
    def __getitem__(self, idx):
        sample = {'image': self.normalize(self.images[idx].float()/255),
                  'pose': self.poses[idx],
                  'target': self._target_generator(self.poses[idx])[0]}
        return sample

    @property
    def joint_pairs(self):
        #Joint pairs which defines the pairs of joint to be swapped
        # when the image is flipped horizontally.
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]