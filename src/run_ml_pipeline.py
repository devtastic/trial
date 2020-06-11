"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/dev/Documents/github/nd320-c3-3d-imaging-starter/section2/src/clean_data/TrainingSet"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "."

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
    # Output: np array {"image": image, "seg": label, "filename": f}
#     print('Data shape in run file after loading data: ', data.shape)
#     print('Data[1-image; 45-axial] shape in run file after loading data: ', data[1]["image"].shape)
#     print('Data[1-image; 45-axial, 1] shape in run file after loading data: ', data[1]["image"][0].shape)

# #     print('Data[seg] shape in run file after loading data: ', data[1].shape)
# #     print(data[0])
#     slice_test = []
#     for i, d in enumerate(data):
# #         print('***In for loop***')
# #         print(i)
# #         print(d)
# #         print('d["image"].shape[0]: ', d["image"].shape[0])
#         for j in range(d["image"].shape[0]):
#             slice_test.append((i, j))#(1,1) (1,2)
#         if(i == 2): break
    
# #     for i,j in slice_test:
# #         print(f'i: {i} and j: {j}')
# #     print('slice_test: ', slice_test)#[(0,0),.. (0,36), (1,0).. (1,30),(2,0)... (2,41)]
#     idx = 45
#     print('slice_test.shape: ', len(slice_test))#110    
#     slc = slice_test[idx]
#     print('slc: ', slc)#(1,8)
#     print('slc[0]: ', slc[0])#1 - points at a file/image
#     print('slc[1]: ', slc[1])#8 - points at axial pixel
    
#     sample = dict()
#     sample["id"] = idx
    
# #     print('data[slc[0]]: ', data[slc[0]])#Points to an record in data with image, seg and filename
    
#     print('data[slc[0]]["image"]: ', data[slc[0]]["image"].shape)
    
    # a = data[slc[0]]["image"]             #Image shape: (31,64,64)
    # b = a[0].shape   #(1, 64, 64)
    # print(b)
    # print("***")
    # print(a)
    # print("***")
    # print(a[0])
    
#     d = np.array([
#                     [
#                         [1, 2, 3, 0, 0], 
#                         [4, 5, 6, 0, 0]
#                     ], 
#                     [
#                         [7, 8, 9, 1, 1], 
#                         [10, 11, 12, 1, 1]
#                     ],
#                     [
#                         [13, 14, 15, 2, 2],
#                         [16, 17, 18, 2, 2]  
#                     ]
#                     ])
#     print(d) #(3,2,5)
#     print(d.shape)
#     print(d[0])
#     print(d[0][0])
#     print(d[0][0][0])
    
#     a = data[slc[0]]["image"][slc[1]]
#     print(a)
    
#     b = data[slc[0]]["seg"][slc[1]]
#     print(b)

    
    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    split["train"] = keys[0:184]
    split["val"] = keys[184:236]
    split["test"] = keys[236:]

    # Set up and run experiment
    
#     # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    # exp.load_model_parameters('/home/dev/Documents/github/nd320-c3-3d-imaging-starter/section2/src/2020-06-10_1318_Basic_unet/model.pth')
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

