import sys
import tensorflow as tf
import glob
import os

from capsa import MVEWrapper, EnsembleWrapper
from models import unet
from losses import MSE
import matplotlib.pyplot as plt
import numpy as np
import config
import keras
import h5py
import json
import PIL
import gradio

config.BS = 1
config.N_TRAIN = 1*3


def notebook_select_gpu(idx, quite=True):
    # https://www.tensorflow.org/guide/gpu#using_a_single_gpu_on_a_multi-gpu_system

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[idx], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            if not quite:
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def load_depth_data_train():
    train = h5py.File('/data/capsa/data/depth_train.h5', 'r')
    return (train['image'], train['depth'])


def totensor_and_normalize(x, y):
    x = tf.convert_to_tensor(x, tf.float32)
    y = tf.convert_to_tensor(y, tf.float32)
    return x / 255., y / 255.


def _get_ds(x, y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(x.shape[0])
    ds = ds.batch(config.BS)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_normalized_ds(x, y, shuffle=True):
    x, y = totensor_and_normalize(x, y)
    return _get_ds(x, y, shuffle)


def get_train_datasets():
    (x_train, y_train) = load_depth_data_train()
    ds_train = get_normalized_ds(x_train[:config.N_TRAIN], y_train[:config.N_TRAIN])
    return ds_train


def select_best_checkpoint(model_path):
    checkpoints_path = os.path.join(model_path, 'checkpoints')
    model_name = model_path.split('/')[-2]

    l = sorted(glob.glob(os.path.join(checkpoints_path, '*.tf*')))

    l_split = [float(i.split('/')[-1].split('vloss')[0]) for i in l]

    # select lowest loss
    min_loss = min(l_split)
    # represent same model
    model_paths = [i for i in l if str(min_loss) in i]
    path = model_paths[0].split('.tf')[0]
    return f'{path}.tf', model_name


def load_model(path, model_name, ds, opts={'num_members':3}, quiet=True):
    if model_name in ['mve', 'notebook_mve']:
        their_model = unet()
        model = MVEWrapper(their_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    elif model_name in ['ensemble', 'notebook_ensemble']:
        num_members = opts['num_members']

        their_model = unet()
        model = EnsembleWrapper(their_model, num_members=num_members)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LR),
            loss=MSE,
        )

    else:
        return NotImplementedError

    print("******* Train to load ******* ")
    # https://github.com/tensorflow/tensorflow/issues/33150#issuecomment-574517363
    _ = model.fit(ds, epochs=1, verbose=1)
    load_status = model.load_weights(path)


    # base mode tires to load optimizer as well, so load_status gives error
    # if model_name not in ['base', 'notebook_base']:
    #     # used as validation that all variable values have been restored from the checkpoint
    #     load_status.assert_consumed()
    if not quiet:
        print(f'Successfully loaded weights from {path}.')
    return model


def write_json(content, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(content))


def read_json(file_name):
    with open(file_name) as f:
        content = json.loads(f.read())
    return content


def predict(inp):
    # Preprocess the data
    inp_resized = inp.resize((160, 128))
    ## convert to array & change shape to match model input
    inp_np = np.reshape(np.array(inp_resized), (1, 128, 160, 3))
    ## convert to tensor and normalize
    inp_tf = tf.convert_to_tensor(inp_np, tf.float32) / 255.

    # load model
    best_checkpoints_path = "best_checkpoints.json"
    model_name_aleatoric = 'mve'
    checkpoints_dict = read_json(best_checkpoints_path)
    small_train_ds = get_train_datasets()
    trained_aleatoric = load_model(checkpoints_dict['aleatoric'], model_name_aleatoric, small_train_ds, quiet=False)

    # run model on the input
    pred = trained_aleatoric(inp_tf)
    # convert pred to numpy
    y_hat, aleatoric, epistemic, bias = pred.numpy()

    # transfer to -1 to 1 range
    y_hat_clipped = (np.clip(y_hat, a_min=0, a_max=1)*2 - 1)
    print("y_hat min = {}, max = {}".format(np.amax(y_hat_clipped), np.amin(y_hat_clipped)))
    y_hat_img_np = y_hat_clipped[0, :, :, 0]

    # aleatoric = aleatoric*255/np.amax(aleatoric)
    aleatoric_img_np = aleatoric[0, :, :, 0]
    # transfer to -1 to 1 range
    alea_min = np.amin(aleatoric_img_np)
    alea_max = np.amax(aleatoric_img_np)
    print("alea min {} and max {} before transfer".format(alea_min, alea_max))
    aleatoric_img_np = -1 + (aleatoric_img_np-alea_min)/(alea_max-alea_min) * 2
    print("alea min {} and max {} before transfer".format(np.amin(aleatoric_img_np), np.amax(aleatoric_img_np)))

    return (inp_np[0], y_hat_img_np, aleatoric_img_np)


def main():
    best_checkpoints_path = "best_checkpoints.json"
    model_name_aleatoric = 'mve'
    img_shape = (128, 160)
    # ALEATORIC_PATH = '/data/capsa/depth/results/mve/20220827-211807'
    # EPISTEMIC_PATH = '/data/capsa/depth/results/ensemble/20220829-172414-3members'
    #
    # # notebook_select_gpu(0)
    #
    # dummy_data = create_dummy_data()
    #
    # path_aleatoric, _ = select_best_checkpoint(ALEATORIC_PATH)
    #
    # # model_name_aleatoric = 'mve'
    # # print("Aleatoric: path: {}, model_name: {}".format(path_aleatoric, model_name_aleatoric))
    # # trained_aleatoric = load_model(path_aleatoric, model_name_aleatoric, dummy_data, quiet=False)
    #
    # path_epistemic, _ = select_best_checkpoint(EPISTEMIC_PATH)
    # # model_name_epistemic = 'ensemble'
    # # print("Epistemic: path: {}, model_name: {}".format(path_epistemic, model_name_epistemic))
    # # trained_epistemic = load_model(path_epistemic, model_name_epistemic, dummy_data, quiet=False)
    #
    # checkpoint_paths = {'aleatoric': path_aleatoric,
    #                     'epistemic': path_epistemic}
    # print("checkpoint_paths: ", checkpoint_paths)
    # write_json(checkpoint_paths, 'best_checkpoints.json')

    # # -------------- Load the model -----------------
    # checkpoints_dict = read_json(best_checkpoints_path)
    # small_train_ds = get_train_datasets()
    # trained_aleatoric = load_model(checkpoints_dict['aleatoric'], model_name_aleatoric, small_train_ds, quiet=False)
    #
    # # -------------- test the model on an image ----------------- TODO: FIX the bug
    # # read the image
    # test_img = PIL.Image.open("test_1.jpeg")
    # img_resized = test_img.resize((128, 160))
    # # convert to array
    # # change shape to match model input
    # img_np = np.reshape(np.array(img_resized), (1, 128, 160, 3))
    # # img_np = np.array(img_resized)
    #
    # # convert to tensor and normalize
    #
    # img_tf = tf.convert_to_tensor(img_np, tf.float32) / 255.
    #
    # pred = trained_aleatoric(img_tf)
    # y_hat, aleatorics, _, _ = pred.numpy()
    #
    # #clipping y_hat to 0 and 1
    # y_hat_clipped = np.clip(y_hat, a_min=0, a_max=1)
    # cmap = plt.cm.jet
    # cmap.set_bad(color='black')
    # plt.imsave("test_1_pred.jpeg", y_hat_clipped[0, :, :, 0], cmap=cmap)
    # plt.imsave("test_1_uncert.jpeg", aleatorics[0, :, :, 0], cmap=cmap)



    # -------------- Now that we can run prediction for each image lets setup Gradio ----------------
    gradio.Interface(fn=predict,
                     inputs=gradio.Image(type="pil"),
                     outputs=[gradio.Image(), gradio.Image(),
                              gradio.Image()],
                     examples=["test_1.jpeg"]).launch(share=True)


if __name__ == "__main__":
    main()