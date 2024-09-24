import numpy as np
import matplotlib.pyplot as plt

# import tensorflow as tf
import h5py

#  Tracks, ECAL, HCAL
f3 = h5py.File("../../data/quark-gluon/quark-gluon_train-set_n793900.hdf5","r")
f2 = h5py.File("../../data/quark-gluon/quark-gluon_test-set_n10000.hdf5","r")
f = h5py.File("../../data/quark-gluon/quark-gluon_test-set_n139306.hdf5","r")
x_train = f3.get('X_jets')
y_train = f3.get('y')

x_val = f2.get('X')
y_val = f2.get('y')

x_test = f2.get('X')
y_test = f2.get('y')
x_val_ones = x_val[y_val[()]==1]
x_val = x_val[y_val[()]==0]

div1 = np.max(x_val, axis=(1,2)).reshape((x_val.shape[0],1,1,3))
div1[div1 == 0] = 1
x_val = x_val / div1
div2 = np.max(x_val_ones, axis=(1,2)).reshape((x_val_ones.shape[0],1,1,3))
div2[div2 == 0] = 1
x_val_ones = x_val_ones / div2

x_test = x_val
x_test_ones = x_val_ones

def crop(x, channel, crop_fraction):
    return f.image.central_crop(x[:,:,:,channel].reshape(x.shape[0],125,125,1), crop_fraction)
def crop_and_resize(x, channel, scale, crop_fraction=0.8,meth="bilinear"):
    cropped = tf.image.central_crop(x[:,:,:,channel].reshape(x.shape[0],125,125,1), crop_fraction)
    return tf.image.resize(cropped, (scale,scale), method=meth).numpy()
def simple_resize(x, channel, scale, meth="bilinear"):
    return tf.image.resize(x[:,:,:,channel].reshape((x.shape[0],125,125,1)), (scale,scale), method=meth).numpy()
batch_size = 20
num_batches = x_train.shape[0]//batch_size

events = num_batches*batch_size
fnew = h5py.File("QG_train_normalized", "w")
dsetx = fnew.create_dataset("X", (events,125,125,3), dtype='f')
dsety = fnew.create_dataset("y", (events,), dtype='i')
 


for i in range(int(num_batches)):
    y = y_train[i * batch_size: (i + 1) * batch_size]
    x = x_train[i * batch_size: (i + 1) * batch_size]

    div1 = np.max(x, axis=(1,2)).reshape((x.shape[0],1,1,3))
    div1[div1 == 0] = 1
    x = x / div1

    dsety[i * batch_size: (i + 1) * batch_size] = y
    dsetx[i * batch_size: (i + 1) * batch_size] = x
    print("batch ",i,"/",num_batches, end="\r")
    
fnew.close()

def load_qg_img(electron_file, photon_file, reduced_dim=None, dataset_size=-1, channel=0):
    """
    Load and preprocess electron and photon data.

    Args:
        electron_file (str): Path to the electron data file.
        photon_file (str): Path to the photon data file.
        reduced_dim (int): Size to resize the images to (default is None).
        dataset_size (tuple): Custom dataset size (train_size, val_size, test_size) for faster training (default is None).

    Returns:
        dict: A dictionary with the preprocessed training, validation, and test datasets.
    """
    f_electron = h5py.File(electron_file, "r")
    f_photon = h5py.File(photon_file, "r")

    # print(f_electron['X'].shape, f_photon['X'].shape)
    # print(f_electron['y'].shape, f_photon['y'].shape)
    
    electrons = f_electron['X'][:, :, :, channel][:dataset_size]
    photons = f_photon['X'][:, :, :, channel][:dataset_size]
    electrons_y = f_electron['y'][:][:dataset_size]
    photons_y = f_photon['y'][:][:dataset_size]

    x = np.vstack((electrons, photons))
    y = np.hstack((electrons_y, photons_y))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=False)

    if reduced_dim:
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((reduced_dim, reduced_dim), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])
        x_train = np.stack([resize_transform(img) for img in x_train])
        x_val = np.stack([resize_transform(img) for img in x_val])
        x_test = np.stack([resize_transform(img) for img in x_test])

    train_dataset = ParticleDataset(x_train, y_train)
    val_dataset = ParticleDataset(x_val, y_val)
    test_dataset = ParticleDataset(x_test, y_test)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }
