from collections import Counter
import random

def gini_impurity(y):
    total = len(y)

    count_0 = y.count(0)
    count_1 = y.count(1)

    p_0 = count_0 / total
    p_1 = count_1 / total

    return 1 - (p_0 ** 2) - (p_1 ** 2)


def gini_split(y_left, y_right):
    total = len(y_left) + len(y_right)

    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)

    weighted_gini = (
        (gini_left * len(y_left)) / total
        + (gini_right * len(y_right)) / total
    )

    return weighted_gini


def is_pure(y):
    return len(set(y)) == 1


def majority_class(y):
    return Counter(y).most_common(1)[0][0]


def split_data(X, y, feature_index, threshold):
    left_X = []
    right_X = []
    left_y = []
    right_y = []

    for i in range(len(X)):
        if X[i][feature_index] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    return left_X, right_X, left_y, right_y


def find_best_split(X, y, max_features):
    best_gini = float("inf")
    best_threshold = None
    best_feature = None

    n_features = len(X[0])

    selected_features = random.sample(
        range(n_features),
        min(max_features, n_features)
    )

    for feature in selected_features:
        values = sorted(set(row[feature] for row in X))

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            left_X, right_X, left_y, right_y = split_data(
                X, y, feature, threshold
            )

            if not left_y or not right_y:
                continue

            current_gini = gini_split(left_y, right_y)

            if current_gini < best_gini:
                best_gini = current_gini
                best_threshold = threshold
                best_feature = feature

    return best_feature, best_threshold, best_gini

def build_tree(X, y, max_features, depth=0, max_depth=5):

    if is_pure(y):
        return {"prediction": y[0]}

    if depth >= max_depth:
        return {"prediction": majority_class(y)}

    best_feature, best_threshold, best_gini = find_best_split(
        X, y, max_features
    )

    if best_feature is None:
        return {"prediction": majority_class(y)}

    left_X, right_X, left_y, right_y = split_data(
        X, y, best_feature, best_threshold
    )

    left_branch = build_tree(
        left_X, left_y, max_features, depth + 1, max_depth
    )

    right_branch = build_tree(
        right_X, right_y, max_features, depth + 1, max_depth
    )

    return {
        "feature_index": best_feature,
        "threshold": best_threshold,
        "left": left_branch,
        "right": right_branch
    }

def predict(tree, x):

    if "prediction" in tree:
        return tree["prediction"]

    feature_index = tree["feature_index"]
    threshold = tree["threshold"]

    if x[feature_index] <= threshold:
        return predict(tree["left"], x)

    return predict(tree["right"], x)

def create_bootstrap_sample(X,y):
    indices = random.choices(range(len(X)), k=len(X))

    X_sample = [X[i] for i in indices]
    y_sample = [y[i] for i in indices]

    return X_sample,y_sample


def train_random_forest(X, y, n_trees=5, max_features=2, max_depth=5):

    forest = []

    for i in range(n_trees):

        X_sample, y_sample = create_bootstrap_sample(X, y)

        tree = build_tree(
            X_sample,
            y_sample,
            max_features,
            max_depth=max_depth
        )

        forest.append(tree)

    return forest

def predict_random_forest(forest, x):

    predictions = []

    for tree in forest:
        prediction = predict(tree, x)
        predictions.append(prediction)

    final_prediction = Counter(predictions).most_common(1)[0][0]

    return final_prediction

X = [
    [10, 2000, 3],
    [20, 3000, 4],
    [30, 4000, 5],
    [40, 5000, 2],
    [50, 6000, 3],
    [60, 7000, 4]
]

y = [0, 0, 1, 1, 1, 0]

forest = train_random_forest(
    X,
    y,
    n_trees=5,
    max_features=2,
    max_depth=5
)

new_data = [25, 3500, 4]

result = predict_random_forest(forest, new_data)

print("Random Forest Prediction:", result)