from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load data and shaping label
    data = load_iris().data
    labels = load_iris().target
    labels = np.reshape(labels,(150,1))
    data = np.concatenate([data, labels], axis=-1)


    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
    dataset = pd.DataFrame(data, columns=names)
    dataset['species'].replace(0, 'Iris-setosa', inplace=True)
    dataset['species'].replace(1, 'Iris-versicolor', inplace=True)
    dataset['species'].replace(2, 'Iris-virginica', inplace=True)
    dataset.head(5)

    # Visualize the data
    plt.figure(4, figsize=(10, 8))
    plt.scatter(data[:50, 0], data[:50, 1], c='r', label='Iris-setosa')
    plt.scatter(data[50:100, 0], data[50:100, 1], c='g', label='Iris-versicolor')
    plt.scatter(data[100:, 0], data[100:, 1], c='b', label='Iris-virginica')
    plt.xlabel('Sepal length', fontsize=20)
    plt.ylabel('Sepal width', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Sepal length vs. Sepal width', fontsize=20)
    plt.legend(prop={'size': 18})
    # plt.show()

    plt.figure(4, figsize=(8, 8))
    plt.scatter(data[:50, 2], data[:50, 3], c='r', label='Iris-setosa')
    plt.scatter(data[50:100, 2], data[50:100, 3], c='g', label='Iris-versicolor')
    plt.scatter(data[100:, 2], data[100:, 3], c='b', label='Iris-virginica')
    plt.xlabel('Petal length', fontsize=15)
    plt.ylabel('Petal width', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Petal length vs. Petal width', fontsize=15)
    plt.legend(prop={'size': 20})
    # plt.show()

    # Insight label covriance and correlation
    dataset.iloc[:50, :].corr()  # setosa
    dataset.describe()
    train_data, test_data, train_label, test_label = train_test_split(dataset.iloc[:, :3], dataset.iloc[:, 4],
                                                                      test_size=0.2, random_state=42)

    # Apply classifier
    neighbors = np.arange(1, 9)
    train_accuracy = np.zeros(len(neighbors))
    test_accuracy = np.zeros(len(neighbors))

    # Cross validate with each labels quantity
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_label)
        train_accuracy[i] = knn.score(train_data, train_label)
        test_accuracy[i] = knn.score(test_data, test_label)

    # Evaluation training progress
    plt.figure(figsize=(10, 6))
    plt.title('KNN accuracy with varying number of neighbors', fontsize=20)
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend(prop={'size': 20})
    plt.xlabel('Number of neighbors', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()

    # Train with neighbor 3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_label)
    train_accuracy = knn.score(train_data, train_label)
    test_accuracy = knn.score(test_data, test_label)

    prediction = knn.predict(test_data)
    for item, predict in zip(test_label, prediction):
        # print(item, predict)
        print('{:<25s} {:<26s} {:1}'.format(item, predict, item==predict))
        # print("Location: %s  Revision: %s" % (item,predict))
    # print(prediction, )
    # print(type(test_label))
    # print(len(test_label))
