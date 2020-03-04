from __future__ import print_function, division
import os, sys
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
import json
from random import shuffle




class Data(Dataset):
    """
    gesture frame dataset.
    """

    def __init__(self, root_dir, partition, transform=None):
        """
        Args:
            root_dir (string): Directory with all the h5 sequences.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform
    def __len__(self):
        return len(self.partition)
    def __getitem__(self, index):
        ''' get an item in the data, either a whole sequence or a single frame
         returns 2 tensors, the sequence(or frame) and the labels
         '''
        data, sequence,num= [], "", None
        if type(self.partition[index]) is list:# if it's a single frame we have [name of the sequence , number the frame]
            sequence, num = self.partition[index][0], self.partition[index][1]
        else: # else it's just the name of the sequence
            sequence = self.partition[index]
        sequence = os.path.join(self.root_dir, sequence+ ".h5") # the name of the file
        for i in range(4):# all the dimensions
            with h5py.File(sequence, 'r') as f:
                if num!= None: # if there is a frame number , meaning that it's frame by frame
                    data.append(f['ch{}'.format(i)][num]) #we return the frame/ its label
                    label = f['label'][num]
                else:# if it's a whole sequence
                    data.append(f['ch{}'.format(i)][()])
                    label = f['label'][()]
        data = np.stack((data[0], data[1], data[2], data[3]), axis=-1) # make it a 4d image, works wether sequence or frame
        if self.transform: 
            data, label = self.transform((data, label))# make the necessary transformations
        return data, label
        
    def get(self, gesture , session, instance):
        # function to retrieve a sequence 
        return self.__getitem__(str(gesture) + "_" + str(session) + "_"+ str(instance))

            
class Reshape(object):
    '''
    reshape an image to 32 * 32 
    '''
    def __call__(self, sample):
        data, label = sample
        if len(data.shape) == 3:# a whole sequence 
            data = data.reshape((data.shape[0], 32,32, 4))
        else: # a single frame
            data = data.reshape((32,32,4))
        return data, label
class Rescale(object):
    
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple): Desired output size. If tuple, output is
            matched to output_size. 
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        newH, newW = self.output_size
        data, label = sample[0], sample[1]
        if len(data.shape) == 4: # if sequence
            data = transform.resize(data, (data.shape[0], newH, newW, data.shape[3]))
        else:# if single frame
            data = transform.resize(data, (newH, newW, data.shape[2]))

        return data, label


class ToTensor(object):
    """transposes ndarrays in sample to Tensors.
        and transforms them to tensors
    """

    def __call__(self, sample):
        data, label= sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(data.shape) == 4:# a whole sequence 
            data = data.transpose((0, 3, 1, 2))
        else: # a single frame
            data = data.transpose((2,0,1))
        return torch.from_numpy(data),torch.from_numpy(label)

def split(root_dir, frames = False, already_defined = False, percentage = 0.5,
              use = 1):
        """
        args : 
            root_dir : directory where the h5 files are stored
            frames: boolean, returns sets of (sequence, image number) if true, sets of sequences if false
            already_defined : returns the predefined train set if true, random if false
            percentage : percentage of the data in the train set
            use : percentage of used data, set to 100% by default, only applicable for already_defined set to False
        return train and test sets
            
        """
        if not frames and already_defined:
            train, test = [], []
            with open("../partitions/file_half.json") as f:#get the defined train set 
                train = json.load(f)["train"]
            for i in os.listdir(root_dir): # put the rest of the files in the test set
                if i[:-3] not in train:
                    test.append(i[:-3])
            return train, test
        elif not frames and not already_defined:
            data = list(map(lambda x : x[:-3] , os.listdir(root_dir)))#take all the sequences
            shuffle(data)
            return data[: int(percentage * len(data) * use)],data[int(percentage * len(data)*use):int(len(data)*use)]
        elif frames:
            if not os.path.exists('../partitions/all_frames.json'):#if the list of all the image isn't created
                print("building the frames index")
                all_frames = []
                for i in os.listdir(root_dir):
                    # we go through all the sequences and create a list of all [sequence,frame]
                    with h5py.File(root_dir+ i, 'r') as f:
                        length = len(f['ch{}'.format(0)][()])
                        all_frames.extend([ (i[:-3],j) for j in  list(range(length))])
                with open('../partitions/all_frames.json', 'w') as outfile:
                    json.dump({"data" : all_frames}, outfile)# we dump it in a json file 
            
            #open the set of frames, return the train test plit
            data = []
            with open('../partitions/all_frames.json', 'r') as infile:
                data = json.load(infile)["data"]
            shuffle(data)
            return data[: int(percentage * len(data)*use)],data[int(percentage * len(data)*use): int(len(data)*use)]
        

