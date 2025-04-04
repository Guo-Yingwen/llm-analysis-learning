import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature      # 特征
            self.threshold = threshold  # 阈值
            self.left = left           # 左子树
            self.right = right         # 右子树
            self.value = value         # 叶节点的值

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件：达到最大深度，样本数量太少，或者类别已经纯净
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < 2 or n_classes == 1:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # 寻找最佳分割点
        best_feature = None
        best_threshold = None
        best_gain = -1

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain == -1:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # 根据最佳分割点分割数据
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return self.Node(best_feature, best_threshold, left, right)

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    # 创建并训练决策树
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    # 进行预测
    X_test = np.array([[2, 3], [6, 7]])
    predictions = clf.predict(X_test)
    print("预测结果:", predictions) 