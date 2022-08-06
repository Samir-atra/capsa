import numpy as np
import h5py
f = h5py.File('../../data/nyu_depth_v2_labeled.mat','r')
data = f.get('images')
data = np.array(data) # shape = (1449, 3, 640, 480)
data = data.reshape((1449, 640, 480, 3))
print(data.shape)
np.save("training_data_labeled.npy", data)

depths = f.get('depths')
depths = np.array(depths) # shape = (1449, 3, 640, 480)
np.save("depths_labeled.npy", depths)
