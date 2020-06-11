"""
Contains class that runs inferencing
"""
import torch
import numpy as np
import pickle

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """

        patch_size = 64
        volume = med_reshape(volume, new_shape=(volume.shape[0], patch_size, patch_size))

        return self.single_volume_inference(volume)
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        print('volume shape: ', volume.shape) 
        # print('volume[0] shape: ', volume[0].shape) 
        # print('volume[0:1] shape: ', volume[0:1].shape)
        
        # for i in range(volume.shape[0]):
        #     slices.append(volume[0:1])

        # for i in range(volume.shape[0]):
        #     slices.append(i)
 
        #slices.append(volume[0:1])
        # arr = np.zeros(volume.shape, dtype=np.float32) 
        mask3d = np.zeros(volume.shape)

        # trial = volume[11, :, :]
        # a = torch.from_numpy(np.asarray(trial).astype(np.single) / 0xff).unsqueeze(0).unsqueeze(0)
        # prediction = self.model(a.to(self.device))
        # print(prediction.shape)

        # mask3d[slc_idx, :, :] = torch.argmax(prediction, dim = 0)

        for slc_idx in range(volume.shape[0]):
            sliced = volume[slc_idx, :, :]
            a = torch.from_numpy(sliced.astype(np.single) / 0xff).unsqueeze(0).unsqueeze(0)
            prediction = self.model(a.to(self.device))
            # print(prediction.shape)

            mask3d[slc_idx, :, :] = torch.argmax(np.squeeze(prediction.cpu()), dim = 0)
            # slices.append(prediction.argmax(1).cpu().numpy().squeeze())
            # slices.append(torch.argmax(np.squeeze(prediction.cpu()), dim = 0))

            # prediction_softmax = F.softmax(prediction, dim=1)
            # print(prediction_softmax)
        
        # print('len(slices): ', len(slices))
        # print('slices[0]: ', slices[0])
        
        # a = torch.from_numpy(slices).type(torch.FloatTensor).unsqueeze(0)
        # print(a.shape)
            
        # for idx, label in enumerate(slices):
        #     print(f'idx: {idx}')
        #     print(f'label: {label}')
            # print(label)
#             if label is not np.nan:
            #label = label.split(" ")
            #mask = np.zeros(volume.shape[1] * volume.shape[2], dtype=np.uint8)

            #posit = map(int, label[0::2])
            #leng = map(int, label[1::2])

            #for p, l in zip(posit, leng):
            #    mask[p:(p+l)] = 1
           # arr[:, :, idx] = mask.reshape(volume.shape[1], volume.shape[1], order='F')

        
        # slices = np.asarray(slices)
        # slices.reshape((-1, slices.shape[0], slices.shape[1]))
        return mask3d #slices #np.array(slices)
