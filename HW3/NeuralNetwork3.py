import numpy as np
import pandas as pd
import layers


def predict_label(f):
    return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


def read_csv(train_image_file, test_image_file, train_label_file, test_label_file):
    """
    Read data files
    :param train_image_file:
    :param test_image_file:
    :param train_label_file:
    :param test_label_file:
    :return:
    """
    train_image = np.array(pd.read_csv(train_image_file, header=None))
    test_image = np.array(pd.read_csv(test_image_file, header=None))
    train_label = np.array(pd.read_csv(train_label_file, header=None))
    test_label = np.array(pd.read_csv(test_label_file, header=None))
    return train_image, test_image, train_label, test_label


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
    test_image_file = "Data/test_image.csv"
    train_image_file = "Data/train_image.csv"
    test_label_file = "Data/test_label.csv"
    train_label_file = "Data/train_label.csv"

    # Hyperparameters
    batch_size = 100
    dropout_rate = 0.2
    num_L1 = 784
    num_L2 = 10
    learning_rate = 0.001
    num_epoch = 10

    train_image, test_image, train_label, test_label = read_csv(train_image_file, test_image_file, train_label_file, test_label_file)
    X_train, X_valid, y_train, y_valid = train_valid_split(train_image, train_label, test_size=0.1)

    N_train = X_train.shape[0]
    dimensions = X_train.shape[1]

    # Model
    model = dict()
    model['L1'] = layers.Dense(input_D=num_L1, output_D=num_L2)
    model['relu1'] = layers.Relu()
    model['loss'] = layers.SoftmaxCrossEntropy()

    for t in range(num_epoch):
        # Minibatch generate
        idx_permute = np.random.permutation(N_train)
        num_batches = int(N_train // batch_size)

        for i in range(num_batches):
            X_batch, y_batch = get_minibatch(X_train, y_train, idx_permute[i * batch_size: (i + 1) * batch_size])

            a1 = model['L1'].forward(X_batch)
            h1 = model['relu1'].forward(a1)
            loss = model['loss'].forward(h1, y_batch)

            grad_a1 = model['loss'].backward(a1, y_batch)
            grad_h1 = model['relu1'].backward(h1, grad_a1)
            grad_x = model['L1'].backward(X_batch, grad_h1)

            # Update parameters
            for module_name, module in model.items():

                # check if a module has learnable parameters
                if hasattr(module, 'params'):
                    for key, _ in module.params.items():
                        if len(key) == 1:
                            g = module.params[key]
                            module.params[key] -= learning_rate * g

        # Validation accuracy
        val_acc = 0
        a1 = model['L1'].forward(X_valid)
        h1 = model['relu1'].forward(a1)
        val_acc += np.sum(predict_label(h1) == y_valid)

        print("Epoch: {} ".format(t) + "Loss: {} ".format(loss) + "acc: {}".format(val_acc))




