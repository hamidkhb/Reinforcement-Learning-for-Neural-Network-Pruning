import argparse
import client
import deep_network as dn
import tensorflow as tf
import reinforcement_learning as rl




def check_args(args):

    if args.method not in ["UCB1", "KL_UCB", "Bayes_UCB", "TS_Beta", "TS_Normal"]:
        raise ValueError ("given method should be one of the following :" \
                         "UCB1, KL_UCB, Bayes_UCB, TS_Beta, TS_Normal")


def main():

    #creating a connection to cluster
    if args.client:
        c = client.Start_Client(gpu_name=str(args.gpu))


    path = "/home/fe/khodabakhshandeh/Projects/radar/radar-ml/Python/data/Config G/box_data.npy"
    train_data, valid_data, test_data = dn.Load_Data(path)

    if args.load_model == None:

        model = dn.Make_Model()
        print(model.summary())
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        model.fit(train_data, verbose=1, validation_data=valid_data, epochs=25)

    else:

        model = tf.keras.models.load_model(args.load_model)

    print("evaluating network on the whole dataset...")
    print("this accuracy and loss is being used as reference to the performance of pruning")

    data, label = dn.Data_Prep(path)
    model.evaluate(data, label)


    rf.train(model, train_data, args.method, args.horizon)
















if __name__ ==  "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--client", type=bool, default=True, help="True if using the client for cluster is necessary")
    ap.add_argument("--load_model", type=str, default=None,
                    help="the path to the model, which if given stops training new model")
    ap.add_argument("--method", type=str, required=True,
                    help="choosing the RL method. please choose from: UCB1, KL_UCB, Bayes_UCB, TS_Beta, TS_Normal")

    ap.add_argument("--horizon", type=int, default=7000, help="number of rounds to choose a bandit")

    args = ap.parse_args()

    check_args(args)
    main()