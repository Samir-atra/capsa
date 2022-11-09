import os
import pandas as pd
from PIL import Image
import numpy as np

data_path = '/data/biasbounty/split'

def load_split_data():
    label_map = {"gender": {"male": 1, "female": 2}, 
                 "age": {"0_17": 1, "18_30": 2, "31_60": 3, "61_100": 4} 
                 }
    split_data_x = []
    split_data_y = []
    for f in os.listdir(data_path):
        if f == ".DS_Store":
            continue
        # Asumption for the format gender_l-b-age_u-b-age_monk_color-range
        file_name_splits = f.split("_")
        y_gender = label_map["gender"][file_name_splits[0]]
        y_age = label_map["age"][file_name_splits[1] + "_" + file_name_splits[2]]
        y_color = int(file_name_splits[4])
        path_to_imgs = os.path.join(data_path, f)

        print("Reading files in: {}".format(f))

        for img_path in os.listdir(path_to_imgs):
            if img_path == ".DS_Store":
                continue
            x_pil = Image.open(os.path.join(path_to_imgs, img_path))
            x_np = np.array(x_pil.getdata())
            split_data_x.append(x_np)
            split_data_y.append([y_gender, y_age, y_color])
    print(split_data_x)
    split_data_x_np = np.array(split_data_x)
    split_data_y_np = np.array(split_data_y)

    np.save(os.path.join(data_path, "split_data_x.npy"), split_data_x_np)
    np.save(os.path.join(data_path, "split_data_y.npy"), split_data_y_np)

    print("Saved split data in {} & {}".format(os.path.join(data_path, "split_data_x.npy"), os.path.join(data_path, "split_data_y.npy")))


def main():
    load_split_data()


if __name__ == "__main__":
    main()