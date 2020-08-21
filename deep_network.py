import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt

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


def Data_Prep(path: str):
    #preparing data for training
    dataset_raw = np.load(path, allow_pickle=True)
    arrays = np.array(dataset_raw[()]["data"])
    arrays = arrays - arrays.min()
    arrays = arrays / arrays.max()
    arrays -= arrays.mean()
    arrays = arrays / arrays.std()
    if np.isnan(arrays).any() or np.isinf(arrays).any():
        raise ValueError("data have imperfections")
    print(arrays.shape)
    labels = dataset_raw[()]["label"]
    labels = np.array([x - np.array(list(set(labels))).min() for x in labels])
    print(labels.shape)


    return (arrays, labels)


def Load_Data(path : str):

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


def Evaluate_Rl(result: dict, model_path:str, method:str):

    if len(result.keys()) == 2:
        n = result["n"]
        reward = result["mu"]
    elif len(result.keys()) == 3:
        n = result["n"]
        success = result["s"]
        fail = result["f"]
        reward = success/(success+fail)



    n2 = n.reshape(-1)
    mu2 = reward.reshape(-1)
    plt.figure(figsize=(15, 5))
    plt.title("Distribution of visits per node for {} algorithm".format(method))
    plt.ylabel("Number of visits")
    plt.xlabel("weight in the last layer")
    plt.bar(np.arange(len(n2)), n2, )
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(len(mu2)), mu2)
    plt.title("average reward per node for {} algorithm".format(method))
    plt.ylabel("reward")
    plt.xlabel("weight in the last layer")

    #sorting the weights based on best average reward
    indexes = np.unravel_index(np.argsort(reward, axis=None), reward.shape)

    data, label = Data_Prep("/home/fe/khodabakhshandeh/Projects/radar/radar-ml/Python/data/Config G/box_data.npy")

    loss = []
    accuracy = []
    for i in range(160):
        model = tf.keras.models.load_model(model_path)
        model.evaluate(data, label, verbose=0)
        W = model.layers[-1].get_weights()
        W_ = np.copy(W)
        W_[0][indexes[0][-(i + 1):], indexes[1][-(i + 1):]] = 0
        model.layers[-1].set_weights(W_)
        l, acc = model.evaluate(data, label)
        loss.append(l)
        accuracy.append(acc)

    x = np.arange(len(accuracy)) / reward.size * 100
    plt.figure(figsize=(7, 5))
    plt.plot(x, loss)
    plt.title("loss value after pruning based on {} algorithm".format(method))
    plt.ylabel("loss")
    plt.xlabel("percentage of pruned weights")
    plt.grid()
    plt.figure(figsize=(7, 5))
    plt.plot(x, accuracy)
    plt.title("accuracy after pruning based on {} algorithm".format(method))
    plt.ylabel("accuracy")
    plt.xlabel("precentage of pruned weights")
    plt.grid()