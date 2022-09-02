# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/models/depth/deterministic.py

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, \
    UpSampling2D, Cropping2D, concatenate, ZeroPadding2D, SpatialDropout2D
import functools

############# U-net #############

import notebooks.configs.demo as config_demo
from losses import MSE

def unet(input_shape=(128, 160, 3), drop_prob=0.0, reg=None, activation=tf.nn.relu, num_class=1, compile=False):

    concat_axis = 3
    inputs = tf.keras.layers.Input(shape=input_shape)
    # inputs_normalized = tf.multiply(inputs, 1/255.)

    Conv2D_ = functools.partial(Conv2D, activation=activation, padding='same', kernel_regularizer=reg, bias_regularizer=reg)

    conv1 = Conv2D_(32, (3, 3))(inputs)
    conv1 = Conv2D_(32, (3, 3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = SpatialDropout2D(drop_prob)(pool1)

    conv2 = Conv2D_(64, (3, 3))(pool1)
    conv2 = Conv2D_(64, (3, 3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = SpatialDropout2D(drop_prob)(pool2)

    conv3 = Conv2D_(128, (3, 3))(pool2)
    conv3 = Conv2D_(128, (3, 3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = SpatialDropout2D(drop_prob)(pool3)

    conv4 = Conv2D_(256, (3, 3))(pool3)
    conv4 = Conv2D_(256, (3, 3))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = SpatialDropout2D(drop_prob)(pool4)

    conv5 = Conv2D_(512, (3, 3))(pool4)
    conv5 = Conv2D_(512, (3, 3))(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D_(256, (3, 3))(up6)
    conv6 = Conv2D_(256, (3, 3))(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D_(128, (3, 3))(up7)
    conv7 = Conv2D_(128, (3, 3))(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D_(64, (3, 3))(up8)
    conv8 = Conv2D_(64, (3, 3))(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D_(32, (3, 3))(up9)
    conv9 = Conv2D_(32, (3, 3))(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(num_class, (1, 1))(conv9)

    # conv10 = tf.multiply(conv10, 255.)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

    if not compile:
        return model
    # for demo
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config_demo.LR),
            loss=MSE,
        )
        return model

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

# import numpy as np
# their_model = create((128, 160, 3))
# x = np.ones((1, 128, 160, 3), dtype=np.float32)
# output = their_model(x)
# print(output.shape)


############# AutoEncoder #############

Conv2D_ = functools.partial(Conv2D, activation=tf.nn.relu, padding='same', kernel_regularizer=None, bias_regularizer=None)

def get_encoder(input_shape=(128, 160, 3), drop_prob=0.0):

    inputs = tf.keras.layers.Input(shape=input_shape)

    conv1 = Conv2D_(32, (3, 3))(inputs)
    conv1 = Conv2D_(32, (3, 3))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_(64, (3, 3))(pool1)
    conv2 = Conv2D_(64, (3, 3))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_(128, (3, 3))(pool2)
    conv3 = Conv2D_(128, (3, 3))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_(256, (3, 3))(pool3)
    conv4 = Conv2D_(256, (3, 3))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    model = tf.keras.models.Model(inputs=inputs, outputs=pool4, name="encoder")
    return model

def get_bottleneck(input_shape=(8, 10, 256)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    conv5 = Conv2D_(16, (3, 3))(inputs)
    conv5 = Conv2D_(16, (3, 3))(conv5)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv5)
    return model

def get_decoder(input_shape=(8, 10, 16), num_class=3):

    inputs = tf.keras.layers.Input(shape=input_shape)

    up_conv5 = UpSampling2D(size=(2, 2))(inputs)
    conv6 = Conv2D_(256, (3, 3))(up_conv5)
    conv6 = Conv2D_(256, (3, 3))(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Conv2D_(128, (3, 3))(up_conv6)
    conv7 = Conv2D_(128, (3, 3))(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2D_(64, (3, 3))(up_conv7)
    conv8 = Conv2D_(64, (3, 3))(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Conv2D_(32, (3, 3))(up_conv8)
    conv9 = Conv2D_(32, (3, 3))(conv9)

    conv10 = Conv2D(num_class, (1, 1))(conv9)

    model = tf.keras.models.Model(inputs=inputs, outputs=conv10, name="decoder")
    return model

### check dims
# import numpy as np
# x = np.ones((1, 128, 160, 3), dtype=np.float32)
# print('x_train:', x.shape)

# enc = get_encoder((128, 160, 3))
# enc_out = enc(x)
# print('enc_out: ', enc_out.shape)

# bottleneck = get_bottleneck((8, 10, 256))
# bot_out = bottleneck(enc_out)
# print('bot_out: ', bot_out.shape)

# dec = get_decoder((8, 10, 128), num_class=3)
# dec_out = dec(bot_out)
# print('dec_out: ', dec_out.shape)


class AutoEncoder(tf.keras.Model):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.enc = get_encoder((128, 160, 3))
        self.bottleneck = get_bottleneck((8, 10, 256))
        self.dec = get_decoder((8, 10, 16), num_class=3)

    def loss_fn(self, y, y_hat):
        loss = tf.reduce_mean(
            self.compiled_loss(y, y_hat, regularization_losses=self.losses),
        )

        return loss

    def train_step(self, data):
        x, _ = data
        y = x

        with tf.GradientTape() as t:
            # interesting interpretation: vae whose variance is set to 0
            y_hat = self(x, return_risk=False)
            loss = self.loss_fn(y, y_hat)

        # self.compiled_metrics.update_state(y, y_hat)
        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {f'loss': loss}

    def test_step(self, data):
        x, _ = data
        y = x

        y_hat = self(x, return_risk=False)
        loss = self.loss_fn(y, y_hat)
        return {f'loss': loss}

    def call(self, x, return_risk=True):
        h = self.enc(x) # (B, 128, 160, 3) -> (B, 8, 10, 256)
        h = self.bottleneck(h) # (B, 8, 10, 256) -> (B, 8, 10, 16) 
        y_hat = self.dec(h) # (B, 8, 10, 16) -> (B, 128, 160, 3)

        if return_risk:
            epistemic = tf.keras.metrics.mean_squared_error(x, y_hat)
            return (y_hat, tf.expand_dims(epistemic, -1))
        else:
            return y_hat