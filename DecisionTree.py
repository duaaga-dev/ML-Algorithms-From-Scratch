
from collections import Counter


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


def split_data(X, y, threshold):
    left_X = []
    right_X = []
    left_y = []
    right_y = []

    for i in range(len(X)):
        if X[i] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])

    return left_X, right_X, left_y, right_y


def find_best_split(X, y):
    best_gini = float("inf")
    best_threshold = None

    sorted_X = sorted(set(X))

    thresholds = [
        (sorted_X[i] + sorted_X[i + 1]) / 2
        for i in range(len(sorted_X) - 1)
    ]

    for threshold in thresholds:
        left_X, right_X, left_y, right_y = split_data(X, y, threshold)

        if not left_y or not right_y:
            continue

        current_gini = gini_split(left_y, right_y)

        if current_gini < best_gini:
            best_gini = current_gini
            best_threshold = threshold

    return best_threshold, best_gini


def build_tree(X, y):

    if is_pure(y):
        return {"prediction": y[0]}

    threshold, best_gini = find_best_split(X, y)

    if threshold is None:
        return {"prediction": majority_class(y)}

    left_X, right_X, left_y, right_y = split_data(X, y, threshold)

    left_branch = build_tree(left_X, left_y)
    right_branch = build_tree(right_X, right_y)

    return {
        "threshold": threshold,
        "left": left_branch,
        "right": right_branch
    }


X = [1, 2, 3, 4, 5, 6]
y = [0, 0, 0, 1, 1, 1]

tree = build_tree(X, y)

print(tree)

def predict(tree, x):

    # إذا وصلنا إلى Leaf Node
    if "prediction" in tree:
        return tree["prediction"]

    # إذا لم نصل إلى Leaf Node
    if x <= tree["threshold"]:
        return predict(tree["left"], x)
    else:
        return predict(tree["right"], x)

    tree = build_tree(X, y)

    print(predict(tree, 2))  # 0
    print(predict(tree, 5))  # 1
    print(predict(tree, 3))  # 0
    print(predict(tree, 6))  # 1





    X_test = [1.5, 3.5, 4.5, 6]
    predictions = []

    for x in X_test:
     predictions.append(predict(tree, x))

    print(predictions)
