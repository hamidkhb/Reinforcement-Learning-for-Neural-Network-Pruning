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

    if args.model_path == None:

        model = dn.Make_Model()
        print(model.summary())
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
        model.fit(train_data, verbose=1, validation_data=valid_data, epochs=25)

        model.save("model_base.h5")
        args.model_path = "model_base.h5"

    else:

        model = tf.keras.models.load_model(args.model_path)

    print("evaluating network on the whole dataset...")
    print("this accuracy and loss is being used as reference to the performance of pruning")

    data, label = dn.Data_Prep(path)
    model.evaluate(data, label)

    print("starting reinforcement learning ...")
    result = rl.Train(model, train_data, args.method, args.horizon)
    print("evaluating the results ...")
    dn.Evaluate_Rl(result, args.model_path, args.method)

















if __name__ ==  "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="3", help="name of available gpu on cluster")
    ap.add_argument("--client", type=bool, default=True, help="True if using the client for cluster is necessary")
    ap.add_argument("--model_path", type=str, default=None,
                    help="the path to the model, which if given stops training new model")
    ap.add_argument("--method", type=str, required=True,
                    help="choosing the RL method. please choose from: UCB1, KL_UCB, Bayes_UCB, TS_Beta, TS_Normal")

    ap.add_argument("--horizon", type=int, default=7000, help="number of rounds to choose a bandit")

    args = ap.parse_args()

    check_args(args)
    main()