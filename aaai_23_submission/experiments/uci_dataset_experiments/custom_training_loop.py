from hparams import h_params
from sklearn.model_selection import train_test_split
from dataloader import load_dataset
import tensorflow as tf
import numpy as np
from capsa import DropoutWrapper, MVEWrapper, EnsembleWrapper

class Trainer:
    def __init__(self, model, dataset, num_iters = 20000, num_epochs = None, batch_size=128, has_dropout=False):
        self.model = model
        self.num_epochs = num_epochs
        self.num_iters = num_iters
        self.dataset = dataset
        self.dataset_hparams = h_params[dataset]
        self.batch_size = batch_size
        self._prepare_dataset(self.dataset)
        self.min_rmse = float('inf')
        self.min_nll = float('inf')
        self.model_type = type(model)
        self.has_dropout=has_dropout

    def _prepare_dataset(self, dataset):
        (x_train, y_train), (x_test, y_test), scale = load_dataset(dataset)
        if self.num_epochs is None:
            total_train_size = len(x_train)
            self.num_epochs = max(self.num_iters // total_train_size, 1)
            print("num_epochs", self.num_epochs)
            print("length of training set", total_train_size)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = val_dataset.batch(self.batch_size)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.single_test_batch = (x_test[:self.batch_size], y_test[:self.batch_size])
        self.scale = scale

    def evaluate(self, x, y):
        if self.model_type == MVEWrapper:
            outputs = tf.stack([self.model(x, training=True) for _ in range(20)])
            mu_batch = tf.stack([output[0] for output in outputs])
            var_batch = tf.stack([output[1] for output in outputs])

            mean_mu = tf.reduce_mean(mu_batch, axis=0)
            var = tf.reduce_mean(var_batch + mu_batch**2, axis=0) - mean_mu**2
            
            sigma = tf.sqrt(var)
        elif self.model_type == DropoutWrapper:
            mean_mu, sigma = self.model(x, training=True)
        elif self.model_type == EnsembleWrapper:
            if self.has_dropout:
                outputs = tf.stack([self.model(x, training=True) for _ in range(20)])
                mu_batch = tf.stack([output[0] for output in outputs])
                var_batch = tf.stack([output[1] for output in outputs])

                mean_mu = tf.reduce_mean(mu_batch, axis=0)
                var = tf.reduce_mean(var_batch + mu_batch**2, axis=0) - mean_mu**2
                sigma = tf.sqrt(var)
            else:
                mean_mu, sigma = self.model(x, training=True)
        
        rmse = tf.sqrt(tf.reduce_mean((mean_mu-y)**2))

        logprob = -tf.math.log(sigma) - 0.5*tf.math.log(2*np.pi) - 0.5*((y-mean_mu)/sigma)**2
        nll = tf.reduce_mean(-logprob, axis=0)

        return mean_mu, sigma**2, rmse, nll


    def train(self):
        intervals_without_decrease_rmse = 0
        intervals_without_decrease_nll = 0
        for epoch in range(self.num_epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                metrics = self.model.train_step((x_batch_train, y_batch_train))
                
                if step % 100 == 0:
                    x_batch_test, y_batch_test = self.single_test_batch
                    mean_mu, var, rmse, nll = self.evaluate(x_batch_test, y_batch_test)
                    nll += np.log(self.scale[0,0])
                    rmse *= self.scale[0,0]
                    print(metrics)
                    if rmse.numpy() < self.min_rmse:
                        self.min_rmse = rmse.numpy()
                        intervals_without_decrease_rmse = 0
                        print("new min rmse", self.min_rmse)
                    else:
                        intervals_without_decrease_rmse += 1
                        print("rmse hasn't decreased in", intervals_without_decrease_rmse, "cur_steps", self.model.optimizer.iterations)

                    if nll.numpy() < self.min_nll:
                        self.min_nll = nll.numpy()
                        intervals_without_decrease_nll = 0
                        print("new min nll", self.min_nll)
                    else:
                        intervals_without_decrease_nll += 1
                        print("nll hasn't decreased in", intervals_without_decrease_nll, "cur_steps", self.model.optimizer.iterations)

        return self.model, self.min_rmse, self.min_nll