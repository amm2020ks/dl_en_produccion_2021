"""A simple main file to showcase the template."""

import argparse
import logging.config
import os
import time

import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils
from tensorflow.keras import callbacks

from . import __version__


LOGGER = logging.getLogger()
VERSION = __version__


def _download_data():
    LOGGER.info("Downloading data...")
    train, test = datasets.mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    return x_train, y_train, x_test, y_test


def _preprocess_data(x, y, is_convolutional):
    LOGGER.info("Transforming data")
    if is_convolutional == 1:
        x.reshape(-1, 28, 28,1)
    x = x / 255.0
    y = utils.to_categorical(y)
    return x,y


def _build_model(is_convolutional):
    if is_convolutional == 0:
        m = models.Sequential()

        m.add(layers.Input((28,28), name='my_input_layer'))
        m.add(layers.Flatten())
        m.add(layers.Dense(128, activation=activations.relu))
        m.add(layers.Dense(64, activation=activations.relu))
        m.add(layers.Dense(32, activation=activations.relu))
        m.add(layers.Dense(10, activation=activations.softmax))
    elif is_convolutional == 1:
        m = models.Sequential()

        m.add(layers.Input((28, 28, 1), name='my_input_layer'))
        m.add(layers.Conv2D(32, (3, 3), activation=activations.relu))
        m.add(layers.MaxPooling2D((2, 2)))
        m.add(layers.Conv2D(16, (3, 3), activation=activations.relu))
        m.add(layers.MaxPooling2D((2, 2)))
        m.add(layers.Conv2D(8, (3, 3), activation=activations.relu))
        m.add(layers.MaxPooling2D((2, 2)))
        m.add(layers.Flatten())
        m.add(layers.Dense(10, activation=activations.softmax))

    return m


def train_and_evaluate(batch_size, epochs, job_dir, output_path, is_hypertune, is_convolutional):
    # Download the data
    x_train, y_train, x_test, y_test = _download_data()

    # Preprocess the data
    x_train, y_train = _preprocess_data(x_train, y_train, is_convolutional)
    x_test, y_test = _preprocess_data(x_test, y_test, is_convolutional)

    # Build the model
    model = _build_model(is_convolutional)
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])

    # Train the model
    logdir = os.path.join(job_dir, "logs/scalars" + time.strftime("%Y%m%d-%H%M%S"))
    tb_callback = callbacks.TensorBoard(log_dir=logdir)
    model.fit(x_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[tb_callback])

    # Evaluate the model
    loss_value, accuracy = model.evaluate(x_test, y_test)
    LOGGER.info("  *** LOSS VALUE:  %f     ACCURACY: %.4f" % (loss_value, accuracy))

    # Communicate the results of the evaluation of the model
    if is_hypertune:
        metric_tag = 'accuracy_live_class'
        eval_path = os.path.join(job_dir, metric_tag)
        writer = tf.summary.create_file_writer(eval_path)
        with writer.as_default():
            tf.summary.scalar(metric_tag, accuracy, step=epochs)
        writer.flush()

    # Save model in TF SavedModel Format
    if not is_hypertune:
        model_dir = os.path.join(output_path, VERSION)
        models.save_model(model, model_dir, save_format='tf')

def main():
    """Entry point for your module."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypertune', action='store_true', help='This is a hypertuning job')
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training')
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI Platform')
    parser.add_argument('--model-output-path', help='Path to write the SaveModel format')
    parser.add_argument('--convolutional', type=int, help='0 if not conv and 1 if conv')

    args = parser.parse_args()

    is_hypertune = args.hypertune
    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path
    is_convolutional = args.convolutional

    train_and_evaluate(batch_size, epochs, job_dir, output_path, is_hypertune, is_convolutional)

if __name__ == "__main__":
    main()