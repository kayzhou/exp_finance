__author__ = 'Kay Zhou'

'''
用于论文扩充深度学习 RNN
'''
from keras.layers import LSTM, Activation, Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_svmlight_file
from keras.optimizers import Adam, SGD
import numpy as np
from sklearn.model_selection import train_test_split  # 交叉检验
from sklearn import preprocessing


def read_svm_data(in_file):
    X = []
    y = []
    for line in open(in_file):
        temp_X = []
        line = line.strip().split(' ')
        y.append(int(line[0]))
        for i in range(1, len(line)):
            temp_X.append(float(line[i].split(':')[1]))
        X.append(temp_X)
    return X, y


def train_test():
    X, y = read_svm_data(
        '/Users/Kay/Project/EXP/come_on_money/big_board_analysis/data/proportion/SVM_DATA/class3_GOAL4.txt')
    # print(X, y)

    # X_train, X_test = X[: int(len(X) * 4 / 5)], X[int(len(X) * 4 / 5): ]
    # y_train, y_test = y[: int(len(y) * 4 / 5)], y[int(len(y) * 4 / 5): ]
    # print(y_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    y_train = np_utils.to_categorical(y_train, nb_classes=3)
    y_test = np_utils.to_categorical(y_test, nb_classes=3)

    scale = preprocessing.MinMaxScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(-1, 5, 5)
    X_test = X_test.reshape(-1, 5, 5)
    # print(X_train, y_train)
    # print(X_test, y_test)
    # BATCH_SIZE = 32
    TIME_STEPS = 5
    INPUT_SIZE = 5
    OUTPUT_SIZE = 3
    # BATCH_SIZE = 20
    CELL_SIZE = 20
    LR = 0.001

    model = Sequential()

    model.add(LSTM(
        input_shape=(TIME_STEPS, INPUT_SIZE),
        output_dim=CELL_SIZE,
        return_sequences=True,
        # stateful=True,
    ))
    model.add(LSTM(
        input_shape=(TIME_STEPS, INPUT_SIZE),
        output_dim=CELL_SIZE,
    ))

    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    # adam = Adam(LR)

    sgd = SGD(LR)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', 'fmeasure'])

    model.fit(X_train, y_train)
    print('\n验证：')
    print(model.evaluate(X_test, y_test))


if __name__ == '__main__':
    for i in range(10):
        train_test()
