import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """

    return mean_squared_error(targets, predictions)


def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    mlp_regressor = MLPRegressor(max_iter=1000, random_state=0)

    random_search_params = {
        'hidden_layer_sizes': np.logspace(1, 8, 8, base=2).astype(int),
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': np.logspace(-1, -5, 5),
        'learning_rate_init': [1e-3, 1e-4],
        'early_stopping': [True, False]
    }

    rs_cv = RandomizedSearchCV(mlp_regressor, random_search_params, random_state=0)
    rs_cv.fit(X_train, y_train)

    best_model = rs_cv.best_estimator_
    best_params = rs_cv.best_params_

    print(f'The best parameters found by RandomSearchCV are {best_params}.')

    # Calculate predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
