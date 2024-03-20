import numpy as np
from ANN import ANN
from ActivationFunctionFile import ActivationType
import math
import random
from matplotlib import pyplot as plt


class TrigTester:

    def __init__(self):
        """
        creating an ANN for identifying the sin of an angle. Since we'll be taking in values in the range of -2π, +2π,
        and outputting values from -1, 1, we're using the identity activation function for our input and output layers,
        saving the sigmoid for the hidden layers.
        """
        self.myANN = ANN(layer_sizes=[1, 10, 10, 1], activation_ids=[ActivationType.IDENTITY,
                                                                     ActivationType.LEAKY_RELU,
                                                                     ActivationType.LEAKY_RELU,
                                                                     ActivationType.IDENTITY])

    @staticmethod
    def generate_sin_data(N: int, noise: float = 0) -> np.ndarray:
        """
        Generates a set of N number pairs: (ø,sinø), with a random bit of noise in the sin result.
        :param N: the number of pairs requested
        :param noise: a bit of random noise added to the output sine. Default is zero noise.
        :return: an N x 2 numpy array of theta and sin(theta) + random noise, if any.
        """
        result = []
        for count in range(N):
            x = random.random() * 4 * math.pi - 2 * math.pi
            y = math.sin(x) + noise * (random.random() - 0.5)
            result.append((x, y))
        return np.array(result)

    def run_training(self, training_data: np.ndarray):
        """
        Trains the ANN with one cycle of the given data set.
        :param training_data: an N x 2 set of input, output values
        :return: None
        """
        for count in range(training_data.shape[0]):
            self.myANN.predict(np.array(training_data[count][0]))
            expected = np.array((training_data[count][1],))
            self.myANN.backpropagate(expected, alpha=0.0005)

    def perform_N_training_iterations(self, N: int, training_data: np.ndarray):
        """
        Convenience function to perform N cycles of run_training()
        :param N: Number of passes through the training data
        :param training_data: an N x 2 set of input, output values
        :return: None
        """
        for count in range(N):
            # print(f"Training cycle: {i}")
            self.run_training(training_data)

    def rate_performance(self, data_set_to_test: np.ndarray):
        """
        does a prediction for the input of the given data and compares it to the given output data, calculating the
        RMSE (root mean squared error)
        :param data_set_to_test: an M x 2 set of input, output values
        :return: the RMSE for this data set, a float. Ideally, this will be zero.
        """
        sum_squared_error = 0
        for count in range(data_set_to_test.shape[0]):
            xx = data_set_to_test[count][0]
            yy = self.myANN.predict(np.array(xx))
            sum_squared_error += math.pow(yy[0] - data_set_to_test[count][1], 2)
        rms = math.sqrt(sum_squared_error)/(len(data_set_to_test) - 1)
        return rms

    def get_predictions(self, input_data: np.ndarray) -> np.ndarray:
        """
        gets the list of predicted values from the ANN on the input portion of the given data
        :param input_data: a set of M x 2 input, output values (output is ignored).
        :return: a list of corresponding output values.
        """
        output = np.zeros((input_data.shape[0]))
        for count in range(len(input_data)):
            output[count] = self.myANN.predict(input_data[count][0])
        return output


if __name__ == "__main__":
    tt = TrigTester()

    # generate the data to be used for this experiment. The
    train_data: np.ndarray = tt.generate_sin_data(400, noise=0.5)
    validate_data: np.ndarray = tt.generate_sin_data(50)
    test_data: np.ndarray = tt.generate_sin_data(50)

    fig, subplt = plt.subplots(3, 3)

    STEP_SIZE = 40  # Number of training cycles per graph. Each cycle goes through all the data in train_data.
    results = []
    for i in range(8*STEP_SIZE+1):
        if i % STEP_SIZE == 0:
            predictions = tt.get_predictions(validate_data)

            # blue dots - the "actual" data in the training set - x = input;  y = actual
            subplt[int(int(i / STEP_SIZE) / 3)][int(int(i / STEP_SIZE) % 3)].scatter(train_data[:, 0], train_data[:, 1])

            # orange dots - the "predicted" data from the validation set - x = input; y = predicted
            subplt[int(int(i / STEP_SIZE) / 3)][int(int(i / STEP_SIZE) % 3)].scatter(validate_data[:, 0], predictions)

            print(i)  # to get a sense of our progress....

        tt.perform_N_training_iterations(1, train_data)
        if i % int(STEP_SIZE/10) == 0:
            results.append(tt.rate_performance(validate_data))

    plt.show()

    plt.plot(results)
    plt.ylim(bottom=0)
    plt.ylabel("RMS Error")
    plt.xlabel("Training iterations")
    plt.show()

    final_results = tt.rate_performance(test_data)
    print(f"{final_results=}")
