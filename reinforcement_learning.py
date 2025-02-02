import scipy.stats as st
import numpy as np
import deep_network as dn
import tensorflow as tf


def UCB1(mu, n, t):
    #typical upper confidence bound algorithm

    P = np.sqrt(2 * np.log10(t) / n)
    index = np.argmax(np.add(mu, P))

    return index


def KL_UCB(mu:np.array, n:np.array, t:int):
    #Kulback Leibler upper confidence bound

    ub = np.log10(t) / n
    #     print("ub", ub)
    q = np.zeros_like(mu)
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            p = mu[i, j]
            q_tmp = np.arange(p, 1, 0.01)
            d = [_KL_D(p, qi) for qi in q_tmp]
            q[i, j] = d[np.max(np.where(d <= ub[i, j]))]

    index = np.argmax(q)

    return index


def _KL_D(p, q):
    if p == 0:
        return 0
    elif q == 0:
        return np.inf
    else:
        d = p * np.log10(p / q) + (1 - p) * np.log10((1 - p) / (1 - q))
        return d


def TS_Beta(success:int, fail:int):
    #thompson sampling based on beta distribution

    beta = np.random.beta(success + 1, fail + 1)
    index = np.argmax(beta)

    return index


def TS_Normal(mu:np.array, n:np.array):
    #thompson sampling based on gaussian distribution

    t = n + 1
    gaussian = np.random.normal(mu, 1 / t)
    print(gaussian.shape)
    index = np.argmax(gaussian)

    return index


def Bayes_UCB(t:int, success:int, fail:int):
    #bayesian UCB using beta distribution

    d = 1 - 1 / t
    q = st.beta.ppf(d, success + 1, fail + 1)
    index = np.argmax(q)

    return index

def Train(model: tf.keras.Model , dataset , method:str, T:int):

    print("starting training with {} algorithm on {} iterations ...".format(method, T))

    if method in ["Bayes_UCB", "TS_Beta"]:
        W = model.layers[-1].get_weights()
        success = np.zeros((64, 5))
        fail = np.zeros_like(success)
        n = np.zeros_like(success) #number of visits
        threshold = 0.005
        norm_const = 0.03

        for t in range(1, T):
            # select random train data for comparison
            random_data = dn.Select_Random_Data(dataset)

            # selecting index exploration/exploitation
            if np.where(n == 0)[0].size == 0 and np.where(n == 0)[1].size == 0:
                if method == "Bayes_UCB":
                    index = Bayes_UCB(t, success, fail)

                elif method == "TS_Beta":
                    index = TS_Beta(success, fail)

                ind_max = np.array(np.unravel_index(index, success.shape))
                row = ind_max[0]
                col = ind_max[1]

            else:
                #every weight should be visited once to get initial distribution
                row = np.where(n == 0)[0][0]
                col = np.where(n == 0)[1][0]

            print("iteration:", t, "  index:", row, col)

            # evaluating main model
            loss_base = model.evaluate(random_data, verbose=0)[0]

            # setting selected node to zero and evaluating again
            W_ = np.copy(W)
            W_[0][row, col] = 0
            model.layers[-1].set_weights(W_)
            loss = model.evaluate(random_data, verbose=0)[0]

            # calculating delta and reward
            delta = loss_base - loss
            reward = max(0, threshold + delta) / norm_const

            # updating number of successes and fails
            # the threshold for quantization is set to 0.5

            if reward >= 0.5:
                success[row, col] += 1
                print("successful")
            else:
                fail[row, col] += 1
                print("failed")

            n[row, col] += 1

            # initializing the layer to the original trained weights for next round
            model.layers[-1].set_weights(W)

            # saving the results
            file_name = "result_weights_" + method + ".npy"
            results = {"n": n, "s": success, "f": fail}
            np.save(file_name, results)

        return results



    elif method in ["UCB1", "KL_UCB", "TS_Normal"]:

        W = model.layers[-1].get_weights()
        n = np.zeros((64, 5))   # number of visits
        mu = np.zeros_like(n)   # average of reward
        threshold = 0.005
        norm_const = 0.03
        for t in range(1, T):

            # select random train data for comparison
            random_data = dn.Select_Random_Data(dataset)

            # selecting index exploration/exploitation
            if np.where(n == 0)[0].size == 0 and np.where(n == 0)[1].size == 0:
                if method  == "UCB1":
                    index = UCB1(mu, n, t)
                elif method == "KL_UCB":
                    index = KL_UCB(mu, n, t)
                else:
                    index = TS_Normal(mu, n)
                ind_max = np.array(np.unravel_index(index, n.shape))
                row = ind_max[0]
                col = ind_max[1]
            else:
                #every weight should be visited once to get initial distribution
                row = np.where(n == 0)[0][0]
                col = np.where(n == 0)[1][0]

            print("iteration:", t, "  index:", row, col)

            # evaluating main model
            loss_base = model.evaluate(random_data, verbose=0)[0]

            # setting selected node to zero and evaluating again
            W_ = np.copy(W)
            W_[0][row, col] = 0
            model.layers[-1].set_weights(W_)
            loss = model.evaluate(random_data, verbose=0)[0]

            # calculating delta and reward
            delta = loss_base - loss
            reward = max(0, threshold + delta) / norm_const
            # clipping reward
            if reward >= 1:
                reward = 0.99
            print("reward:", reward)

            # updating number of visiting the node and the average reward
            n[row, col] = n[row, col] + 1
            mu[row, col] = ((n[row, col] - 1) / n[row, col]) * mu[row, col] + (1 / n[row, col]) * reward

            # initializing the layer to the original trained weights for next round
            model.layers[-1].set_weights(W)

            #saving the results
            file_name = "result_weights_" + method + ".npy"
            results = {"n": n, "mu": mu}
            np.save(file_name, results)

        return results

