"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []
#         self.slices_labels = []

        for i, d in enumerate(data):#i is the number of element while d contains Image (37,64,64) , Seg and filename
            for j in range(d["image"].shape[0]):#Pick up the axial number (37)
                self.slices.append((i, j))#(0,1) (0,2)... (0,36)
#             for j in range(d["seg"].shape[0]):#Pick up the axial number (37)
#                 self.slices_labels.append((i, j))#(0,1) (0,2)... (0,36)    

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
#         slc_label = self.slices_labels[idx]
        sample = dict()
        sample["id"] = idx

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used
        
        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors. 
        # Your tensor needs to be of shape [1, patch_size, patch_size]
        # Don't forget that you need to put a Torch Tensor into your dictionary element's value
        # Hint: your 3D data sits in self.data variable, the id of the 3D volume from data array
        # and the slice number are in the slc variable. 
        # Hint2: You can use None notation like so: arr[None, :] to add size-1 
        # dimension to a Numpy array
        # <YOUR CODE GOES HERE>
        
#         print(self.data[slc[0]]["filename"])
        
        a = self.data[slc[0]]["image"][slc[1]]
        sample["image"] = torch.from_numpy(a).type(torch.FloatTensor).unsqueeze(0)

        a = self.data[slc[0]]["seg"][slc[1]]
        sample["seg"] = torch.from_numpy(a).type(torch.FloatTensor).unsqueeze(0)
        
        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
