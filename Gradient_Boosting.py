def calculate_mse(y, prediction):
        
 if len(y) == 0:
  return 0
 errors = []


 for actual in y:
  error = (actual - prediction) ** 2
  errors.append(error)

 mse = sum(errors)/ len(errors)
 return mse

y = [10,20,30]
prediction = 20
print(calculate_mse(y,prediction))

def split_data(X, y, threshold):

    X_left = []
    y_left = []

    X_right = []
    y_right = []

    # اكتبي for loop هون
    for i in range(len(X)):
    # إذا X[i] <= threshold
        if X[i] <= threshold:
           X_left.append(X[i])
           y_left.append(y[i])
        else:
            X_right.append(X[i])
            y_right.append(y[i])
    
    return X_left, y_left, X_right, y_right

X = [1, 2, 3, 4]
y = [10, 20, 30, 40]

result = split_data(X, y, 2.5)

print(result)

def weighted_mse(y_left, y_right):

    if len(y_left) == 0 or len(y_right) == 0:
        return float("inf")

    prediction_left = sum(y_left) / len(y_left)
    prediction_right = sum(y_right) / len(y_right)
    total = len(y_left) + len(y_right)

    mse_left = calculate_mse(y_left, prediction_left)
    mse_right = calculate_mse(y_right, prediction_right)

    weighted_mse = (
        (mse_left * len(y_left)) / total
        + (mse_right * len(y_right)) / total
    )

    return weighted_mse

def find_best_split(X, y):

    best_mse = float("inf")
    best_threshold = None

    # 1. احصل على القيم المختلفة والمرتبة من X
    values = sorted(set(X))
    # 2. أنشئ Thresholds بين كل قيمتين متتاليتين
    thresholds = []

    for i in range(len(values)-1):
       threshold = (values[i] + values[i+1]) / 2
       thresholds.append(threshold)
    
    # 3. جرّب كل Threshold باستخدام for loop
    for threshold in thresholds:
       X_left, y_left, X_right, y_right = split_data(X,y,threshold)
       mse = weighted_mse(y_left, y_right)

       if mse < best_mse:
          best_mse=mse
          best_threshold= threshold

    # 6. إذا كان الخطأ أقل من best_mse
    # حدّث best_mse و best_threshold
    return best_threshold, best_mse
best_threshold, best_mse = find_best_split(X, y)

print("Best Threshold:", best_threshold)
print("Best MSE:", best_mse)

def build_tree(X,y,depth,max_depth):
  # 1. Stop condition
  if depth >= max_depth or len(set(X)) < 2 :
        return sum(y) / len(y)
 # 2. Find best split
  best_threshold, best_mse = find_best_split(X,y)
  if best_threshold is None:
    return sum(y) / len(y)

# 3. split data
  X_left, y_left, X_right, y_right = split_data(X, y, best_threshold)
  if len(y_left) == 0 or len(y_right) == 0:
    return sum(y) / len(y)
  
  # 4. Build left and right branches
  left_tree = build_tree(X_left, y_left, depth +1 , max_depth)
  right_tree = build_tree(X_right , y_right, depth+1 , max_depth)

  # 5 return node
  return {
     "threshold" : best_threshold,
     "left": left_tree,
     "right": right_tree
  }

X = [1, 2, 3, 4]
y = [10, 20, 30, 40]

tree = build_tree(X, y, 0, 2)

print(tree)

def predict_tree(tree, x):
     
    if not isinstance(tree, dict):
        return tree

    # 2. إذا كانت x أصغر أو تساوي Threshold
    if x <= tree["threshold"]:
       return predict_tree(tree["left"], x)
  # 3. إذا كانت x أكبر من Threshold
    else:
       return predict_tree(tree["right"], x)

print(predict_tree(tree, 1.2))
print(predict_tree(tree, 2.8))
print(predict_tree(tree, 3.2))