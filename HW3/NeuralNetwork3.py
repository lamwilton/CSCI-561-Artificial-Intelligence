import numpy as np
import pandas as pd
import layers
import sys


def predict_label(a):
    return np.argmax(a, axis=1).reshape((-1, 1))


def read_csv(train_image_file, test_image_file, train_label_file):
    """
    Read data files
    :param train_image_file:
    :param test_image_file:
    :param train_label_file:
    :return:
    """
    train_image = np.array(pd.read_csv(train_image_file, header=None))[0:10000]
    test_image = np.array(pd.read_csv(test_image_file, header=None))
    train_label = np.array(pd.read_csv(train_label_file, header=None))[0:10000]
    return train_image, test_image, train_label


def train_valid_split(train_image, train_label, test_size):
    """
    Train test split as in sklearn
    :param train_image:
    :param train_label:
    :param test_size:
    :return:
    """
    index = int(train_image.shape[0] * (1 - test_size))
    X_train = train_image[0:index, :]
    X_valid = train_image[index:, :]
    y_train = train_label[0:index, :]
    y_valid = train_label[index:, :]
    return X_train, X_valid, y_train, y_valid


def get_minibatch(X_train, y_train, idx):
    X_batch = X_train[idx[0], :]
    y_batch = y_train[idx[0], :]
    for i in range(1, len(idx)):
        X_batch = np.vstack((X_batch, X_train[idx[i], :]))
        y_batch = np.vstack((y_batch, y_train[idx[i], :]))
    return X_batch, y_batch


if __name__ == "__main__":
    # File Paths
    test_image_file = sys.argv[3]
    train_image_file = sys.argv[1]
    train_label_file = sys.argv[2]

    output_file = "test_predictions.csv"

    SCALER = 255

    # Hyperparameters
    test_size = 0.05
    batch_size = 32

    num_L2 = 128
    num_L3 = 10
    learning_rate = 0.001
    num_epoch = 200

    train_image, test_image, train_label = read_csv(train_image_file, test_image_file, train_label_file)

    # Normalizing the data
    train_image = train_image / SCALER
    test_image = test_image / SCALER

    X_train, X_valid, y_train, y_valid = train_valid_split(train_image, train_label, test_size=test_size)

    N_train = X_train.shape[0]
    dimensions = X_train.shape[1]
    num_L1 = dimensions

    # Model
    model = dict()
    model['L1'] = layers.Dense(input_D=num_L1, output_D=num_L2)
    model['relu1'] = layers.Relu()
    model['L2'] = layers.Dense(input_D=num_L2, output_D=num_L3)
    model['loss'] = layers.SoftmaxCrossEntropy()

    for t in range(num_epoch):
        # Minibatch generate
        idx_permute = np.random.permutation(N_train)
        num_batches = int(N_train // batch_size)

        for i in range(num_batches):
            X_batch, y_batch = get_minibatch(X_train, y_train, idx_permute[i * batch_size: (i + 1) * batch_size])

            a1 = model['L1'].forward(X_batch)
            h1 = model['relu1'].forward(a1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y_batch)

            grad_a2 = model['loss'].backward(a2, y_batch)
            grad_h1 = model['L2'].backward(h1, grad_a2)
            grad_a1 = model['relu1'].backward(a1, grad_h1)
            grad_x = model['L1'].backward(X_batch, grad_a1)

            # Update parameters
            for module in [model['L1'], model['L2']]:
                for key in ['W', 'b']:
                    grad = module.params["d" + key]
                    module.params[key] -= learning_rate * grad

        # Validation accuracy
        a1 = model['L1'].forward(X_valid)
        h1 = model['relu1'].forward(a1)
        a2 = model['L2'].forward(h1)
        val_acc = np.sum(predict_label(a2) == y_valid) / X_valid.shape[0]

        print("Epoch: {} ".format(t) + "Loss: {} ".format(loss) + "acc: {}".format(val_acc))

    # Testing
    a1 = model['L1'].forward(test_image)
    h1 = model['relu1'].forward(a1)
    a2 = model['L2'].forward(h1)
    predictions = predict_label(a2).ravel()
    with open(output_file, "w+") as file:
        for item in predictions:
            file.write(str(item))
            file.write("\n")
