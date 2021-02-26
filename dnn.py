import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from scipy.interpolate import interp1d


digits = load_digits()
selection = ''

while not (selection.lower() == '9' or selection.lower() == 'quit'):
    selection = input('Please make a selection from the following: \n'
                      '    1) Train Convolutional Neural Network\n'
                      '    2) Train Neural Network\n'
                      'Note: Please train both deep neural networks before selecting model evaluations.\n'
                      '    3) 5-Fold Cross-validation: CNN\n'
                      '    4) 5-Fold Cross-validation: NN\n'
                      '    5) Confusion Matrix: CNN\n'
                      '    6) Confusion Matrix: NN\n'
                      '    7) ROC Curve: CNN\n'
                      '    8) ROC Curve: NN\n'
                      '    9) Quit\n')

    if selection.lower() == '1':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # re-sizing image to make it suitable for CNN operation
        x_train = x_train.reshape(x_train.shape[0], 8, 8, 1)
        x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)
        input_shape = (8, 8, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # model structure with convolutional layer
        model = Sequential()
        model.add(Conv2D(filters=64,
                         kernel_size=(2, 2),
                         input_shape=input_shape,
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64,
                         kernel_size=(2, 2),
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        # fully connected layers
        model.add(Flatten())
        model.add(Dense(64, activation ='relu'))
        model.add(Dense(64, activation ='relu'))
        model.add(Dense(10, activation='softmax'))

        # training model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3)
        print('\n')

        # evaluating model
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print('Test Loss: ' + str(val_loss) + '\n' + 'Test Accuracy: ' + str(val_acc))
        print('\n')

        # save model
        model.save('CNN_digits.model')

    elif selection.lower() == '2':
        # data-set
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # normalising data-set to aid computation/learning (divides by 255)
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        # model structure
        model = Sequential()
        model.add(tf.keras.layers.Flatten())    # input layer
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        # training model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3)
        print('\n')

        # evaluating model
        val_loss, val_acc = model.evaluate(x_test, y_test)
        print('Test Loss: ' + str(val_loss) + '\n' + 'Test Accuracy: ' + str(val_acc))
        print('\n')

        # save model
        model.save('NN_digits.model')

    elif selection.lower() == '3':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        np.random.seed(1997)
        np.random.shuffle(x)
        np.random.seed(1997)
        np.random.shuffle(y)

        # Split a dataset into k folds
        folds = 5
        x_split = list()
        y_split = list()
        x_copy = list(x)
        y_copy = list(y)
        fold_size = int(len(x) / folds)
        for i in range(folds):
            x_fold = list()
            y_fold = list()
            while len(x_fold) < fold_size:
                x_fold.append(x_copy.pop(0))
                y_fold.append(y_copy.pop(0))
            x_split.append(x_fold)
            y_split.append(y_fold)

        fold_accuracy = 0
        fold_loss = 0
        for i in range(folds):
            x_copy = x_split.copy()
            y_copy = y_split.copy()
            x_test, y_test, x_train, y_train = [], [], [], []

            x_test.append(x_copy.pop(i))  # 1 corresponding split
            y_test.append(y_copy.pop(i))  # 1 corresponding split

            print('Training Cross Validation Fold: ' + str(i + 1) + '\n')

            x_train = []  # 4 corresponding splits
            y_train = []  # 4 corresponding splits

            for sub_list in range(len(x_copy)):
                x_train += x_copy[sub_list]
                y_train += y_copy[sub_list]

            # re-shaping for CNN
            x_train = np.array(x_train).reshape(np.array(x_train).shape[0], 8, 8, -1)
            x_test = np.array(x_test[0]).reshape(np.array(x_test[0]).shape[0], 8, 8, -1)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            y_test = np.asarray(y_test[0])
            y_train = np.asarray(y_train)

            loaded_model = tf.keras.models.load_model('CNN_digits.model')
            loaded_model.fit(x_train, y_train, epochs=3)

            val_loss, val_acc = loaded_model.evaluate(x_test, y_test)

            fold_accuracy += val_acc
            fold_loss += val_loss
        fold_accuracy = round((fold_accuracy / folds), 3)
        fold_loss = round((fold_loss / folds), 3)

        print('\n')
        print('5-Fold Cross Validation Accuracy: ' + str(fold_accuracy) + ' \n5-Fold Loss: ' + str(fold_loss))
        print('\n')

    elif selection.lower() == '4':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        # Split a dataset into k folds
        folds = 5
        np.random.seed(1997)
        np.random.shuffle(x)
        np.random.seed(1997)
        np.random.shuffle(y)

        x_split = list()
        y_split = list()
        x_copy = list(x)
        y_copy = list(y)
        fold_size = int(len(x) / folds)
        for i in range(folds):
            x_fold = list()
            y_fold = list()
            while len(x_fold) < fold_size:
                x_fold.append(x_copy.pop(0))
                y_fold.append(y_copy.pop(0))
            x_split.append(x_fold)
            y_split.append(y_fold)

        fold_accuracy = 0
        fold_loss = 0
        for i in range(folds):
            x_copy = x_split.copy()
            y_copy = y_split.copy()
            x_test, y_test, x_train, y_train = [], [], [], []

            x_test.append(x_copy.pop(i))  # 1 corresponding test split
            y_test.append(y_copy.pop(i))  # 1 corresponding test split

            print('Training Cross Validation Fold: ' + str(i + 1) + '\n')

            x_train = []  # 4 corresponding train splits
            y_train = []  # 4 corresponding train splits

            for sub_list in range(len(x_copy)):
                x_train += x_copy[sub_list]
                y_train += y_copy[sub_list]

            loaded_model = tf.keras.models.load_model('NN_digits.model')
            loaded_model.fit(np.array(x_train), np.array(y_train), epochs=3)
            val_loss, val_acc = loaded_model.evaluate(np.array(x_test[0]), np.array(y_test[0]))

            fold_accuracy += val_acc
            fold_loss += val_loss
        fold_accuracy = round((fold_accuracy / folds), 3)
        fold_loss = round((fold_loss / folds), 3)

        print('\n')
        print('5-Fold Cross Validation Accuracy: ' + str(fold_accuracy) + ' \n5-Fold Loss: ' + str(fold_loss))
        print('\n')

    elif selection.lower() == '5':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        np.random.seed(1)
        indices = np.random.permutation(len(y))
        'retaining 30% of the data-set for testing'
        n_samples = 450

        x_test = digits['data'][indices[-n_samples:]]
        y_test = digits['target'][indices[-n_samples:]]

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)

        loaded_model = tf.keras.models.load_model('CNN_digits.model')
        predictions = loaded_model.predict(x_test)

        confusion_matrix = np.zeros((10, 10), int)
        prediction_list = []
        for i in range(len(y_test)):
            prediction_list.append(np.argmax(predictions[i]))
        for label, prediction in zip(y_test, prediction_list):
            confusion_matrix[label][prediction] += 1

        print(confusion_matrix)
        print('\n')

    elif selection.lower() == '6':
        confusion_matrix = np.zeros((10, 10), int)

        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        np.random.seed(1)
        indices = np.random.permutation(len(y))
        'retaining 30% of the data-set for testing'
        n_samples = 450

        x_test = digits['data'][indices[-n_samples:]]
        y_test = digits['target'][indices[-n_samples:]]
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        loaded_model = tf.keras.models.load_model('NN_digits.model')
        predictions = loaded_model.predict(x_test)

        prediction_list = []
        for i in range(len(x_test)):
            prediction_list.append(np.argmax(predictions[i]))

        for label, prediction in zip(y_test, prediction_list):
            confusion_matrix[label][prediction] += 1
        print(confusion_matrix)
        print('\n')

    elif selection.lower() == '7':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        np.random.seed(1)
        indices = np.random.permutation(len(y))
        'retaining 30% of the data-set for testing'
        n_samples = 450

        x_test = digits['data'][indices[-n_samples:]]
        y_test = digits['target'][indices[-n_samples:]]

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        # re-shaping for CNN
        x_test = x_test.reshape(x_test.shape[0], 8, 8, 1)

        loaded_model = tf.keras.models.load_model('CNN_digits.model')
        predictions = loaded_model.predict(x_test)

        confusion_matrix = np.zeros((10, 10), int)
        prediction_list = []
        for i in range(len(y_test)):
            prediction_list.append(np.argmax(predictions[i]))
        for label, prediction in zip(y_test, prediction_list):
            confusion_matrix[label][prediction] += 1

        # determining ROC values
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR.sort()
        FPR.sort()

        # plotting curve
        plt.title('Receiver Operating Characteristic')
        plt.plot(FPR * (1 / max(FPR)), TPR)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        print('\n')

    elif selection.lower() == '8' or selection.lower() == '':
        x = digits['data'][0:1500]  # FEATURES
        y = digits['target'][0:1500]  # LABELS

        np.random.seed(1)
        indices = np.random.permutation(len(y))
        'retaining 30% of the data-set for testing'
        n_samples = 450

        x_test = digits['data'][indices[-n_samples:]]
        y_test = digits['target'][indices[-n_samples:]]

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        loaded_model = tf.keras.models.load_model('NN_digits.model')
        predictions = loaded_model.predict(x_test)

        confusion_matrix = np.zeros((10, 10), int)
        prediction_list = []
        for i in range(len(y_test)):
            prediction_list.append(np.argmax(predictions[i]))
        for label, prediction in zip(y_test, prediction_list):
            confusion_matrix[label][prediction] += 1

        # determining ROC values
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR.sort()
        FPR.sort()

        print(TPR)
        print(FPR)

        # plotting curve
        plt.title('Receiver Operating Characteristic')
        plt.plot(FPR*(1/max(FPR)), TPR)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        print('\n')
