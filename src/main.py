import numpy as np
import matplotlib.pyplot as plt
from nn_classification import reduce_dimension, train_nn, train_nn_with_regularization, train_nn_with_different_seeds, \
    perform_grid_search
from nn_regression import solve_regression_task, calculate_mse


def task_1_and_2():

    # Load the 'data/features.npy' and 'data/targets.npy' using np.load.
    features = np.load('data/features.npy')
    targets = np.load('data/targets.npy')
    print(f'Shapes: {features.shape}, {targets.shape}')

    __print_images(features)

    features = features.reshape((features.shape[0], -1))
    print(features.shape)

    # PCA
    # Task 1.1.
    print("----- Task 1.1 -----")
    X_reduced = reduce_dimension(features, n_components=120)  # reach at least 85% of the explained variance

    # Task 1.2
    print("----- Task 1.1 -----")
    train_nn(X_reduced, targets)

    # Task 1.3
    print("----- Task 1.3 -----")
    train_nn_with_regularization(X_reduced, targets)

    # Task 1.4
    print("----- Task 1.4 -----")
    train_nn_with_different_seeds(X_reduced, targets)

    # Task 2 - Bonus task. Uncomment the function call if you decide to do this task.
    print("----- Task 2 -----")
    perform_grid_search(X_reduced, targets)


def __print_images(features):
    image_index_list = [260, 900, 1800, 1600, 1400, 2061, 700, 500, 1111, 100]
    for id_img in range(10):
        plt.figure(figsize=(8, 5))
        plt.imshow(features[image_index_list[id_img]])
        plt.axis('off')
        title = "Sign " + str(id_img)
        plt.title(title)
    plt.show()


def task_3():  # Regression with NNs

    # Load 'data/x_datapoints.npy' and 'data/y_datapoints.npy' using np.load.
    x_dataset = np.load('data/x_datapoints.npy')
    y_targets = np.load('data/y_datapoints.npy')
    print(f'Shapes: {x_dataset.shape}, {y_targets.shape}')

    solve_regression_task(x_dataset, y_targets)


def main():
    task_1_and_2()
    task_3()


if __name__ == '__main__':
    main()
