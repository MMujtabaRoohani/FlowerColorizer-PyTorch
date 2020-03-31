import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread
from skimage.transform import resize
import sklearn.neighbors as ne
from sklearn.model_selection import train_test_split
import scipy.misc

from math import sqrt, pi
import time
import os
from os import listdir, walk
from os.path import join, isfile, isdir
import pdb
import random
import sys
import getopt

import torch
from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import connexion
import json
import flask



# Create the application instance
app = connexion.App(__name__, specification_dir='./')
cuda = True if torch.cuda.is_available() else False
# drive.mount('/content/gdrive')
StatePath = "."
DatasetPath = StatePath+"/Nature"

epochs = 100
batch_size = 5
imageSize = 128
learningRate = 0.001
print_freq = 10
save_freq = 10
location = 'cpu'

class NNEncode():
    def __init__(self, NN=5, sigma=5, km_filepath=join(StatePath, 'static', 'pts_in_hull.npy'), train=True, location='cpu'):
        self.cc = np.load(km_filepath)
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = ne.NearestNeighbors(
            n_neighbors=NN, algorithm='ball_tree').fit(self.cc)
        if train:
            self.weights = torch.load(StatePath+'/static/weights.torch')
            if ('cuda' in location):
                self.weights = self.weights.cuda()

    # not in use (too slow) #TODO: make it same as gpu version
    def imgEncode(self, abimg):
        w, h = abimg.shape[1], abimg.shape[2]
        label = torch.zeros((w*h, 313))

        (dists, indexes) = self.nbrs.kneighbors(
            abimg.view(abimg.shape[0], -1).t(), self.NN)
        dists = torch.from_numpy(dists).float()
        indexes = torch.from_numpy(indexes)
        
        weights = torch.exp(-dists**2/(2*self.sigma**2))
        weights = weights/torch.sum(weights, dim=1).view(-1, 1)

        pixel_indexes = torch.Tensor.long(torch.arange(
            start=0, end=abimg.shape[1]*abimg.shape[2])[:, np.newaxis])
        label[pixel_indexes, indexes] = weights
        label = label.t().contiguous().view(313, w, h)

        rebal_indexes = indexes[:, 0]
        rebal_weights = self.weights[rebal_indexes]
        rebal_weights = rebal_weights.view(w, h)
        rebal_label = rebal_weights * label

        return rebal_label

    # computes soft encoding of ground truth ab image, multiplied by weight (for class rebalancing)
    def imgEncodeTorch(self, abimg):
        abimg = abimg.cuda()
        w, h = abimg.shape[1], abimg.shape[2]
        label = torch.zeros((w*h, 313))
        label = label.cuda()

        (dists, indexes) = self.nbrs.kneighbors(
            abimg.view(abimg.shape[0], -1).t(), self.NN)
        dists = torch.from_numpy(dists).float().cuda()
        indexes = torch.from_numpy(indexes).cuda()

        weights = torch.exp(-dists**2/(2*self.sigma**2)).cuda()
        weights = weights/torch.sum(weights, dim=1).view(-1, 1)

        pixel_indexes = torch.Tensor.long(torch.arange(
            start=0, end=abimg.shape[1]*abimg.shape[2])[:, np.newaxis])
        pixel_indexes = pixel_indexes.cuda()
        label[pixel_indexes, indexes] = weights
        label = label.t().contiguous().view(313, w, h)

        rebal_indexes = indexes[:, 0]
        rebal_weights = self.weights[rebal_indexes]
        rebal_weights = rebal_weights.view(w, h)
        rebal_label = rebal_weights * label

        return rebal_label
    def bin2color(self, idx):
        return self.cc[idx]
    def uint_color2tanh_range(img):
        return img / 128.0 - 1.0
    def tanh_range2uint_color(img):
        return (img * 128.0 + 128.0).astype(np.uint8)
    def modelimg2cvimg(img):
        cvimg = np.array(img[0, :, :, :]).transpose(1, 2, 0)
        return tanh_range2uint_color(cvimg)

def sample_image(grayImage, predImage, actualImage, batch, index):
    gen_imgs = np.concatenate((predImage, actualImage), axis=1)
    os.makedirs(StatePath+"/images/"+str(batch), exist_ok=True)
    scipy.misc.imsave(StatePath+"/images/"+str(batch)+"/"+str(index)+'.jpg', gen_imgs)

class CustomImages(Dataset):
    def __init__(self, root, train=True, val=False, color_space='lab', transform=None, test_size=0.1, val_size=0.125, location='cpu'):
        """
            color_space: 'yub' or 'lab'
        """
        self.root_dir = root
        all_files = []
        for r, _, files in walk(self.root_dir):
          for f in files:
            if f.endswith('.jpg'):
              all_files.append(join(r, f))
        train_val_files, test_files = train_test_split(
            all_files, test_size=test_size, random_state=69)
        train_files, val_files = train_test_split(train_val_files,
                                                  test_size=val_size, random_state=69)
        if (train and val):
            self.filenames = val_files
        elif train:
            self.filenames = train_files
        else:
            self.filenames = test_files

        self.color_space = color_space
        if (self.color_space not in ['rgb', 'lab']):
            raise(NotImplementedError)
        self.transform = transform
        self.location = location
        self.nnenc = NNEncode(location=self.location)
        self.train = train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = imread(self.filenames[idx])
        if self.color_space == 'lab':
            img = rgb2lab(img)
        if self.transform is not None:
            img = self.transform(img)
        bwimg = img[:, :, 0:1].transpose(2, 0, 1)
        bwimg = torch.from_numpy(bwimg).float()
        abimg = img[:, :, 1:].transpose(2, 0, 1)    # abimg dim: 2, h, w
        abimg = torch.from_numpy(abimg).float()
        label = -1
        if (self.train):
            if ('cuda' in self.location):
                label = self.nnenc.imgEncodeTorch(abimg)
            else:
                label = self.nnenc.imgEncode(abimg)
        return (bwimg, label, abimg)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = resize(image, (new_h, new_w))[:self.output_size, :self.output_size, :]
        return img

class MultinomialCELoss(nn.Module):
    def __init__(self):
        super(MultinomialCELoss, self).__init__()

    # x dim: n, q, h, w
    # y dim: n, q, h, w
    # n number of cases
    # h, w height width
    # q number of bins
    # output: loss, as a float
    def forward(self, x, y):
        # softmax 
        # x = torch.exp(x)
        # x_sum = x.sum(1)
        # x_sum = x_sum.view(x_sum.shape[0],1,x_sum.shape[1],x_sum.shape[2])
        # x = x / x_sum
        x = x + 1e-8
        x = torch.log(x)
        zlogz = y*x
        loss = - zlogz.sum()
        loss /= (x.shape[0] * x.shape[2] * x.shape[3])
        return loss

class ColorfulColorizer(nn.Module):
    def __init__(self):
        super(ColorfulColorizer, self).__init__()

        self.op_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.op_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.op_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.op_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.op_8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 313, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.op_9 = nn.Sequential(
            nn.Softmax(dim=1)
        )
        self.op_1.apply(self.init_weights)
        self.op_2.apply(self.init_weights)
        self.op_3.apply(self.init_weights)
        self.op_4.apply(self.init_weights)
        self.op_5.apply(self.init_weights)
        self.op_6.apply(self.init_weights)
        self.op_7.apply(self.init_weights)
        self.op_8.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.op_1(x)
        out = self.op_2(out)
        out = self.op_3(out)
        out = self.op_4(out)
        out = self.op_5(out)
        out = self.op_6(out)
        out = self.op_7(out)
        out = self.op_8(out)
        out = self.op_9(out)
        return out

rescale = Rescale(imageSize)

encoder = ColorfulColorizer()
encoder.load_state_dict(torch.load('./colorizer.pkl', map_location="cpu"))
if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    encoder.cuda()

encoder.eval()
T = 0.38
q = 313  # number of colours
nnenc = NNEncode()
bin_index = np.arange(q)
ab_list = nnenc.bin2color(bin_index)   # q, 2

def startImport(assets):
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """
    
    # Saving to a path
    assets.save('image.png')
    
    # reading image from path
    img = imread('image.png')
    
    # converting to lab space
    img = rgb2lab(img)

    #rescaling to fit nn input size
    img = rescale(img)

    bwimg = img[:, :, 0:1].transpose(2, 0, 1)
    bwimg = torch.from_numpy(bwimg).float()

    abimg = img[:, :, 1:].transpose(2, 0, 1)    # abimg dim: 2, h, w
    abimg = torch.from_numpy(abimg).float()
    bwimg = bwimg.unsqueeze(0)
    output = -1
    with torch.no_grad():
        if 'cuda' in location:
          bwimg = bwimg.cuda()
          abimg = abimg.cuda()
        output = encoder(bwimg)

    l_layer = bwimg.data[0].cpu().numpy()
    bin_probabilities = output.data[0].cpu().numpy()  # bin_probabilities dim: q, h, w
    ab_label = abimg.data.cpu().numpy().astype('float64')

    # convert bin_probab -> ab_pred
    bin_probabilities = np.exp(np.log(bin_probabilities)/T)
    bin_sum = bin_probabilities.sum(0)
    bin_sum = bin_sum.reshape((1, bin_sum.shape[0], bin_sum.shape[1]))
    bin_probabilities /= bin_sum

    # ab_pred dim: 2, h, w
    ab_pred = (bin_probabilities[:, np.newaxis, :, :] * ab_list[:, :, np.newaxis, np.newaxis]).sum(0)

    # img_input = l_layer[0]
#     img_input = np.concatenate((l_layer, torch.zeros([2,128,128])), axis=0)
    img_pred = np.concatenate((l_layer, ab_pred), axis=0)
    # img_actual = np.concatenate((l_layer, ab_label), axis=0)
    
#     img_input = lab2rgb(img_input.transpose(1, 2, 0))
    img_pred = lab2rgb(img_pred.transpose(1, 2, 0))
    # img_actual = lab2rgb(img_actual.transpose(1, 2, 0))
    os.makedirs(StatePath+"/images", exist_ok=True)
    
    scipy.misc.imsave(StatePath+"/images/output.jpg", img_pred)

    return flask.send_from_directory('images', 'output.jpg')
    # return "Hello World"

# Read the swagger.yml file to configure the endpoints
app.add_api('swagger.yml')


# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
