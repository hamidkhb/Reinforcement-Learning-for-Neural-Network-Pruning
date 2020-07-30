import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten


def Split_Dataset(dataset: tf.data.Dataset, validation_data_fraction: float):

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 >= validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 < validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset


def Data_Prep(path):
    #preparing data for training
    dataset_raw = np.load(path, allow_pickle=True)
    arrays = np.array(dataset_raw[()]["data"])
    arrays = arrays - arrays.min()
    arrays = arrays / arrays.max()
    arrays -= arrays.mean()
    arrays = arrays / arrays.std()
    if np.isnan(arrays).any() or np.isinf(arrays).any():
        raise "data have imperfections"
    print(arrays.shape)
    labels = dataset_raw[()]["label"]
    labels = np.array([x - np.array(list(set(labels))).min() for x in labels])
    print(labels.shape)


    return (arrays, labels)


def Load_Data(path):

    #loading data and creating tensorflow dataset
    data, label = Data_Prep(path)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(100000)
    train_dataset, rest = Split_Dataset(dataset, 0.3)
    test_dataset, valid_dataset = Split_Dataset(rest, 0.5)
    train_data = train_dataset.shuffle(1000).batch(10)
    valid_data = valid_dataset.batch(10)
    test_data = test_dataset.batch(10)


    return train_data, valid_data, test_data


def Make_Model():
    #creating deep model for classification

    model = tf.keras.Sequential([
        Input((1, 30, 30)),
        Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu", name="c1",
               data_format="channels_first"),
        Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c2",
               data_format="channels_first"),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same", name="m1", data_format="channels_first"),

        Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", name="c3",
               data_format="channels_first"),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same", name="m2", data_format="channels_first"),

        Flatten(),
        Dense(64, activation="relu", use_bias=True),
        Dense(5, use_bias=True)])

    return model


def Select_Random_Data(train_data):
    train_data.shuffle(1000)
    _, data = Split_Dataset(train_data, 0.1)
    return data