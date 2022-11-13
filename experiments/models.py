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
    conv5 = Conv2D_(4, (3, 3))(inputs)
    conv5 = Conv2D_(4, (3, 3))(conv5)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv5)
    return model

def get_bottleneck_flat(input_shape=(8, 10, 256), is_reshape=True):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs) # (8 * 10 * 256 = 20480, 320)
    x = tf.keras.layers.Dense(320, activation='relu')(x) # (B, 320)
    x = tf.keras.layers.Dense(320, activation='relu')(x) # (B, 320)
    out = tf.keras.layers.Dense(320, activation='relu')(x) # (B, 320)
    if is_reshape:
        out = tf.keras.layers.Reshape((8, 10, 4))(out)
    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model

def get_decoder(input_shape=(8, 10, 4), num_class=3):

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

def get_vae_encoder(input_shape=(128, 160, 3), is_reshape=False):

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
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # (B, 8, 10, 256)

    ### bottleneck
    x = tf.keras.layers.Flatten()(pool4) # (B, 8 * 10 * 256) -> (B, 20480)
    x = tf.keras.layers.Dense(80, activation='relu')(x) # (B, 80)
    x = tf.keras.layers.Dense(80, activation='relu')(x) # (B, 80)
    out = tf.keras.layers.Dense(80, activation='relu')(x) # (B, 80)
    if is_reshape:
        out = tf.keras.layers.Reshape((8, 10, 1))(out)

    model = tf.keras.models.Model(inputs=inputs, outputs=out)
    return model

def get_vae_decoder(input_shape=(40), num_class=3):

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Dense(80, activation=None)(inputs) # (B, 40) -> (B, 80)
    x = tf.keras.layers.Reshape((8, 10, 1))(x)

    up_conv5 = UpSampling2D(size=(2, 2))(x)
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
        # self.bottleneck = get_bottleneck((8, 10, 256))
        self.bottleneck = get_bottleneck_flat((8, 10, 256)) # todo-high: note
        self.dec = get_decoder((8, 10, 4), num_class=3)

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
        h = self.bottleneck(h) # (B, 8, 10, 256) -> (B, 8, 10, 4) 
        y_hat = self.dec(h) # (B, 8, 10, 4) -> (B, 128, 160, 3)

        if return_risk:
            epistemic = tf.keras.metrics.mean_squared_error(x, y_hat)
            return (y_hat, tf.expand_dims(epistemic, -1))
        else:
            return y_hat


class VAE(tf.keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        self.enc = get_encoder((128, 160, 3)) # (B, 8, 10, 256)
        self.bottleneck = get_bottleneck_flat((8, 10, 256), is_reshape=False) #(B, 8, 10, 4)

        # after sampling (using both out_mu and out_logvar) z has ch 4
        self.out_mu = tf.keras.layers.Dense(320, activation=None) # (B, 8, 10, 4)
        self.out_logvar = tf.keras.layers.Dense(320, activation=None) # (B, 8, 10, 4)

        self.dec = get_decoder((8, 10, 4), num_class=3) # (B, 128, 160, 3)

    @staticmethod
    def reparameterize(z_mean, z_log_var, training):
        # noise -- normal multivariate gaussian distribution with 0 mean and identity covariance matrix
        # used in re-parameterization trick
        if training:
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        else:
            # if we're not in the train loop don't do sampling, just return mu (the best value that the encoder can give)
            return z_mean

    def loss_fn(self, y, y_hat, z_mean, z_log_var):
        # reconstruction_loss = tf.reduce_mean(
        #     tf.reduce_sum((y - y_hat)**2, axis=[1, 2])
        # )
        # kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.math.square(y - y_hat), axis=-1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.math.square(z_mean) - tf.math.square(tf.math.exp(z_log_var)),
            axis=-1,
        )

        return reconstruction_loss + kl_loss

    def train_step(self, data):
        x, _ = data
        y = x

        with tf.GradientTape() as t:
            y_hat, z_mean, z_log_var = self(x, training=True)
            loss = self.loss_fn(y, y_hat, z_mean, z_log_var)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {f'loss': loss}

    def test_step(self, data):
        x, _ = data
        y = x

        y_hat, z_mean, z_log_var = self(x, training=False)
        loss = self.loss_fn(y, y_hat, z_mean, z_log_var)
        return {f'loss': loss}

    def call(self, x, training, return_risk=True):
        enc_out = self.enc(x) # (B, 128, 160, 3) -> (B, 8, 10, 256)
        bottleneck_out = self.bottleneck(enc_out) # (B, 8, 10, 256) -> (B, 320)

        z_mean = self.out_mu(bottleneck_out) # (B, 320) -> (B, 320)
        z_log_var = self.out_logvar(bottleneck_out) # (B, 320) -> (B, 320)

        z_ = self.reparameterize(z_mean, z_log_var, training) # (B, 320) and (B, 320) -> (B, 320)
        B = tf.shape(z_)[0]
        z = tf.reshape(z_, [B, 8, 10, 4]) # (B, 320) -> (B, 8, 10, 4)

        y_hat = self.dec(z) # (B, 8, 10, 4) -> (B, 128, 160, 3)

        # print('enc_out: ', enc_out.shape)
        # print('bottleneck_out: ', bottleneck_out.shape)
        # print('mu_out: ', mu.shape)
        # print('logstd_out: ', logstd.shape)
        # print('z: ', z.shape)
        # print('dec_out: ', y_hat.shape)

        assert bottleneck_out.shape == z_.shape

        if return_risk:
            return y_hat, z_mean, z_log_var
        else:
            return y_hat

# todo-high:
# Generating a few samples
# N = 16
# z = torch.randn((N, d)).to(device)
# sample = model.decoder(z)
# display_images(None, sample, N // 4, count=True)