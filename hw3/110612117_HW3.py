# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(seq):
    unique_classes, frequency_count = np.unique(seq, return_counts=True)
    pi = frequency_count / len(seq)
    return 1 - np.sum(pi ** 2)

# This function computes the entropy of a label array.
def entropy(seq):
    unique_classes, frequency_count = np.unique(seq, return_counts=True)
    pi = frequency_count / len(seq)
    return -np.sum(pi * np.log2(pi))

# Node class for the decision tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, prediction_class=None):
        self.prediction_class = prediction_class
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

# Decision tree classifier class
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
        self.feature_usage = {}

    def impurity(self, y, weight):
        if self.criterion == 'gini':
            return self.gini(y, weight)
        elif self.criterion == 'entropy':
            return self.entropy(y, weight)

    def gini(self, y, weight):
        if weight is None:
            weight = np.ones(len(y)) / len(y)
        if len(np.unique(y)) <= 1 or len(y) < 1:
            return 0
        type_zero_target = weight[y == 0]
        type_one_target = weight[y == 1]
        target_type_sum = np.array([np.sum(type_zero_target), np.sum(type_one_target)])
        pi = target_type_sum / np.sum(weight)
        return 1 - np.sum(pi ** 2)

    def entropy(self, y, weight):
        if weight is None:
            weight = np.ones(len(y)) / len(y)
        if len(np.unique(y)) <= 1 or len(y) < 1:
            return 0
        type_zero_target = weight[y == 0]
        type_one_target = weight[y == 1]
        target_type_sum = np.array([np.sum(type_zero_target), np.sum(type_one_target)])
        pi = target_type_sum / np.sum(weight)
        return -np.sum(pi * np.log2(pi))

    # Recursive function to build the decision tree
    def generate_tree(self, X, y, depth, weight):
        unique_classes, count = np.unique(y, return_counts=True)
        if len(unique_classes) == 1:
            return Node(prediction_class=unique_classes[0])
        if (self.max_depth is not None and depth == self.max_depth):
            return Node(prediction_class=unique_classes[np.argmax(count)])
        num_features = X.shape[1]
        best_feature, best_threshold, best_impurity = None, None, float('inf')
        #print("\nbest_impurity start: ", best_impurity)
        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])                         
            for threshold in thresholds:                                                #find all the possible division value of that characteristic, i.e finding all the unique value in that the characteristics
                left_mask = X[:, feature_index] <= threshold
                left_impurity = self.impurity(y[left_mask], weight[left_mask])          #generate two boolean array to mark true for data that has its feature_index feature <= threshold 
                right_mask = ~left_mask                                                 #~to get the opposite boolean array to mark true for data that has its feature_index feature > threshold 
                right_impurity = self.impurity(y[right_mask], weight[right_mask])
                weighted_impurity = (len(y[left_mask]) * left_impurity + len(y[right_mask]) * right_impurity) / float(len(y))
                # print("feature_index = ", feature_index)
                # print("threshold = ", threshold)
                # print("left_impurity = ", left_impurity)
                # print("right_impurity = ", right_impurity)
                # print("weighted_impurity = ", weighted_impurity)
                # print("best_impurity = ", best_impurity)

                #os.system("pause")
                # print("left_impurity", len(y[left_mask]) * left_impurity)
                # print("\nright_impurity", len(y[right_mask]) * right_impurity)
                # print("\nweighted_impurity" , weighted_impurity)
                #print("\nbest_impurity : ", best_impurity)
                if weighted_impurity < best_impurity:
                    best_impurity, best_feature, best_threshold = weighted_impurity, feature_index, threshold
                 # if best_impurity == 0.0:
                    #     print("best_feature = ", best_feature)
                    #     print("best_threshold = ", best_threshold)
                    #     os.system("pause")
                    # print("weighted_impurity" , weighted_impurity)
                    # print("\nNew update!\nbest_impurity : ", best_impurity)
        # If no split improves impurity, create a leaf node
        if best_impurity == self.impurity(y, weight):                                           #if al the possible division won't make the impurity smaller => stop and return current Node
            return Node(prediction_class=unique_classes[np.argmax(count)])
        if best_feature is not None:
            self.feature_usage[best_feature] = self.feature_usage.get(best_feature, 0) + 1      #return 0 if it's the first time being the division characteristic
        left_mask = X[:, best_feature] <= best_threshold                                        #build new decision node and its descendents
        leftNode = self.generate_tree(X[left_mask], y[left_mask], depth + 1, weight[left_mask])
        right_mask = ~left_mask
        rightNode = self.generate_tree(X[right_mask], y[right_mask], depth + 1, weight[right_mask])
        return Node(feature_index=best_feature, threshold=best_threshold, left=leftNode, right=rightNode)

    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y, weight=None):
        if weight is None:
            weight = np.ones(len(y)) / len(y)
        self.tree = self.generate_tree(X, y, depth=0, weight=weight)

    # Recursive function to make predictions
    def predict_tree(self, node, x):
        if node.prediction_class is not None:
            return node.prediction_class
        elif x[node.feature_index] <= node.threshold:
            return self.predict_tree(node.left, x)
        else:
            return self.predict_tree(node.right, x)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = np.array([self.predict_tree(self.tree, x) for x in X])
        return predictions

    # This function plots the feature importance of the decision tree.
    # Inside the DecisionTree class
    def plot_feature_importance_img(self, columns):
        if not self.feature_usage:
            print("Decision tree has not been trained yet.")
            return

        feature_count = [self.feature_usage.get(i, 0) for i in range(len(columns))]
        plt.barh(columns, feature_count)
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        plt.title('Decision Tree Feature Importance')
        plt.show()

        

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=20000):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.weakClassifierWeight = list()  # Store the weight of each weak learner
        self.weakClassifier = list()  # Store the weak learners (decision trees)

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        samplesNum, featuresNum = X.shape
        w = np.ones(samplesNum) / samplesNum
        for classifier in range(self.n_estimators):
            self.weakClassifier.append(DecisionTree(criterion='gini', max_depth=1))
            self.weakClassifier[classifier].fit(X, y, w)
            class_pred = self.weakClassifier[classifier].predict(X)
            errorRate = np.sum(w[class_pred != y])
            if errorRate >= 1.0:
                alpha = 0.0
            elif errorRate <= 0.25:
                alpha = 200
            else:
                alpha = 0.5 * np.log((1 - errorRate) / errorRate)
            InCorrectAwareness = np.where(y - class_pred != 0, -1, 1)
            w *= np.exp(-alpha * InCorrectAwareness)
            w /= np.sum(w)
            self.weakClassifierWeight.append(alpha)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = np.zeros(len(X))
        for i in range(len(self.weakClassifier)):
            y_pred = self.weakClassifier[i].predict(X)
            y_pred[y_pred == 0] = -1
            pred = pred + self.weakClassifierWeight[i] * y_pred
        pred = np.sign(pred)
        pred[pred <= 0] = 0
        return pred
    # def predict(self, X):
    #     pred = np.sum([self.weakClassifierWeight[i] * np.sign(self.weakClassifier[i].predict(X)) for i in range(len(self.weakClassifier))], axis=0)
    #     return np.maximum(pred, 0)


# Do not modify the main function architecture.
# You can only modify the value of the random seed and the arguments of your Adaboost class.
if __name__ == "__main__":
    # Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Set random seed to make sure you get the same result every time.
    # You can change the random seed if you want to.
    np.random.seed(0)

    # Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(columns=["age", "sex", "cp", "fbs", "thalach", "thal"])

    # AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=100)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
