import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings


RANDOM_STATE = 42


warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    pca = PCA(n_components, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(features)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f'Explained variance: {explained_var}')
    return X_reduced


def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [5, 100, 200]

    for num_neurons in n_hidden_neurons:
        print(f'MLP - num of hidden neurons {num_neurons}:')
        mlp = MLPClassifier(max_iter=500, random_state=0, hidden_layer_sizes=num_neurons)
        mlp.fit(X_train, y_train)

        train_acc = mlp.score(X_train, y_train)
        test_acc = mlp.score(X_test, y_test)
        loss = mlp.loss_

        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        print(f'Num training epochs: {mlp.n_iter_}')
        print('---------------------------------------------------------------')


def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    alpha = [1.0, 1e-4, 1.0]
    early_stopping = [False, True, True]
    n_hidden_neurons = [5, 100, 200]

    for model_number in range(len(n_hidden_neurons)):
        for hyperparameter_number in range(len(alpha)):
            print(f'MLP - num hidden = {n_hidden_neurons[model_number]}, alpha = {alpha[hyperparameter_number]}, early_stopping = {early_stopping[hyperparameter_number]}:')
            mlp = MLPClassifier(max_iter=500, random_state=0, hidden_layer_sizes=n_hidden_neurons[model_number], alpha=alpha[hyperparameter_number], early_stopping=early_stopping[hyperparameter_number])
            mlp.fit(X_train, y_train)

            train_acc = mlp.score(X_train, y_train)
            test_acc = mlp.score(X_test, y_test)
            loss = mlp.loss_

            print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
            print(f'Loss: {loss:.4f}')
            print(f'Num training epochs: {mlp.n_iter_}')
            print('---------------------------------------------------------------')


def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    seeds = [8, 43, 18, 64, 97]
    alpha = 1.0
    early_stopping = False
    n_hidden_neurons = 200

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))
    train_losses = np.zeros(len(seeds))
    loss_curves = []
    test_predictions = []

    for seed_number in range(len(seeds)):
        print(f'MLP - num hidden = {n_hidden_neurons}, alpha = {alpha}, early_stopping = {early_stopping}, seed = {seeds[seed_number]}:')
        mlp = MLPClassifier(max_iter=500, random_state=seeds[seed_number], hidden_layer_sizes=n_hidden_neurons, alpha=alpha, early_stopping=early_stopping)
        mlp.fit(X_train, y_train)

        loss_curves.append(mlp.loss_curve_)
        test_predictions.append(mlp.predict(X_test))
        train_acc_arr[seed_number] = mlp.score(X_train, y_train)
        test_acc_arr[seed_number] = mlp.score(X_test, y_test)
        train_losses[seed_number] = mlp.loss_

        print(f'Train accuracy: {train_acc_arr[seed_number]:.4f}. Test accuracy: {test_acc_arr[seed_number]:.4f}')
        print(f'Loss: {train_losses[seed_number]:.4f}')
        print(f'Num training epochs: {mlp.n_iter_}')
        print('---------------------------------------------------------------')

    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    print(f'Training set: min_acc = {np.min(train_acc_arr):.4f}, max_acc = {np.max(train_acc_arr):.4f}')
    print(f'Test set: min_acc = {np.min(test_acc_arr):.4f}, max_acc = {np.max(test_acc_arr):.4f}')

    plt.plot(np.linspace(1, len(loss_curves[0]), len(loss_curves[0])), loss_curves[0])  # use 1st model: seed = 8
    plt.title('loss curve (seed = 8)')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    print("Predicting on the test set (using 1st model: seed = 8)")
    print(confusion_matrix(y_test, test_predictions[0], labels=range(10)))
    print(classification_report(y_test, test_predictions[0], labels=range(10)))

    print('xyx')


def perform_grid_search(features, targets):
    """
    BONUS task: Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    param_grid = {
        "alpha": [0.0, 0.001, 1.0],
        "activation": ['identity', 'logistic', 'relu'],
        "solver": ['lbfgs', 'adam'],
        "hidden_layer_sizes": [100, 200]
    }

    mlp = MLPClassifier(max_iter=500, random_state=0, early_stopping=True)

    grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    scores = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = grid_search.best_estimator_.score(X_test, y_test)

    print(f'The best score is {best_score}.')
    print(f'The best parameters found by GridSearchCV are {best_params}.')
    print(f'The accuracy score on the test set using the best estimator is {test_score}.')
