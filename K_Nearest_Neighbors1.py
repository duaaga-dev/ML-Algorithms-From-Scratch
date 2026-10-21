from collections import Counter


def knn_predict(X_train, Y_train, new_student, K):
    if K < 1 or K > len(X_train):
        raise ValueError("K must be between 1 and the number of training samples")

    neighbors = []

    for x, y in zip(X_train, Y_train):
        distance = abs(new_student - x)
        neighbors.append((distance, y))

    neighbors.sort()

    k_neighbors = neighbors[:K]

    labels = [label for distance, label in k_neighbors]

    votes = Counter(labels)

    prediction = votes.most_common(1)[0][0]

    return prediction


X_train = [1, 2, 4, 5]
Y_train = [0, 0, 1, 1]


X_test = [4.5, 1.5]
Y_test = [1, 0]


K = 3

predictions = []

for x in X_test:
    prediction = knn_predict(X_train, Y_train, x, K)
    predictions.append(prediction)

print("Predictions:", predictions)
print("Actual:", Y_test)


