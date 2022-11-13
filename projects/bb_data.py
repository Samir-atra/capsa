import numpy as np
import pandas as pd
import os, glob
from PIL import Image
from tqdm import tqdm
import pdb

def save_split_data_as_np(data_path='/data/biasbounty/split'):
    split = "/data/biasbounty/split"
    label_file = "/data/biasbounty/train/labels_numeric.csv"

    labels = pd.read_csv(label_file).set_index("name")  # read labels
    all_images = glob.glob(os.path.join(split, "*", "*.png"))  # grab all images
    X_data, Y_data = [], []

    for image_path in tqdm(all_images):  # loop through all images
        # get the image as numpy array
        x = np.array(Image.open(image_path), dtype=np.uint8)

        # check if the image is grayscale, if so either
        # 1. convert to rgb
        # 2. skip it
        if x.shape[-1] != 3:
            print("found faulty data named: {} with shape: {}".format(image_path, x.shape))
            continue

        X_data.append(x)
        # get the labels
        fname = os.path.basename(image_path)
        y = np.array(labels.loc[fname], dtype=int)
        Y_data.append(y)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    np.save(os.path.join("/data/biasbounty/", "split_data_x.npy"), X_data)
    np.save(os.path.join("/data/biasbounty/", "split_data_y.npy"), Y_data)

    print("Saved split data in {} & {}".format(os.path.join(data_path, "split_data_x.npy"), os.path.join(data_path, "split_data_y.npy")))
    pdb.set_trace()


def read_split_data(data_path='/data/biasbounty/'):
    split_x = np.load(os.path.join(data_path, "split_data_x.npy"))
    split_y = np.load(os.path.join(data_path, "split_data_y.npy"))
    return split_x, split_y

def main():
    save_split_data_as_np()


if __name__ == "__main__":
    main()
