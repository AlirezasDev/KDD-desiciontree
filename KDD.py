# Alireza Sepehri - KDD Cup 1999 dataset, preprocesses it by categorizing attack types, selecting top features, discretizing numerical data, and splitting it into train/validation/test sets, then builds, prunes, and evaluates a decision tree to detect network intrusions.

import pandas as pd
import numpy as np
import requests
import gzip
import io
from collections import Counter
from graphviz import Digraph


# Decision Tree Functions
class TreeNode:
    def __init__(self, depth=0, is_leaf=False):
        self.depth = depth
        self.is_leaf = is_leaf
        self.feature = None
        self.value = None
        self.info_gain = None
        self.gini = None
        self.children = {}
        self.label = None

def entropy(y, class_weight=None):
    if len(y) == 0:
        return 0
    counts = Counter(y)
    if class_weight is None:
        probs = np.array([count / len(y) for count in counts.values()])
    else:
        w_counts = [counts.get(cls, 0) * class_weight.get(cls, 1) for cls in class_weight]
        w_total = sum(w_counts) or 1
        probs = np.array([wc / w_total for wc in w_counts])
    return -np.sum(probs * np.log2(probs + 1e-10))

def gini(y, class_weight=None):
    if len(y) == 0:
        return 0
    counts = Counter(y)
    if class_weight is None:
        probs = np.array([count / len(y) for count in counts.values()])
    else:
        w_counts = [counts.get(cls, 0) * class_weight.get(cls, 1) for cls in class_weight]
        w_total = sum(w_counts) or 1
        probs = np.array([wc / w_total for wc in w_counts])
    return 1 - np.sum(probs**2)

def weighted_size(y, class_weight):
    if class_weight is None:
        return len(y)
    counts = Counter(y)
    return sum(counts.get(cls, 0) * class_weight.get(cls, 1) for cls in class_weight)

def information_gain(parent_y, children_y, class_weight=None):
    parent_ent = entropy(parent_y, class_weight)
    weighted_ent = 0
    total_w = weighted_size(parent_y, class_weight) if class_weight else len(parent_y)
    for child in children_y:
        w_child = weighted_size(child, class_weight) if class_weight else len(child)
        weighted_ent += (w_child / total_w) * entropy(child, class_weight)
    return parent_ent - weighted_ent

def best_split(X, y, criterion='entropy', class_weight=None):
    best_gain = -np.inf
    best_feature = None
    best_value = None
    best_gini = np.inf
    for feature in X.columns:
        values = X[feature].unique()
        for value in values:
            left_idx = X[feature] == value
            right_idx = ~left_idx
            left_y, right_y = y[left_idx], y[right_idx]
            if criterion == 'entropy':
                gain = information_gain(y, [left_y, right_y], class_weight)
            else:
                w_left = weighted_size(left_y, class_weight) if class_weight else len(left_y)
                w_right = weighted_size(right_y, class_weight) if class_weight else len(right_y)
                w_total = w_left + w_right
                gain = gini(y, class_weight) - (w_left / w_total * gini(left_y, class_weight) + w_right / w_total * gini(right_y, class_weight))
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value
                best_gini = gini(y, class_weight)
    return best_feature, best_value, best_gain, best_gini

# Dataset Preparation (KDD Cup 1999)
response = requests.get("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz")
if response.status_code == 200:
    with gzip.open(io.BytesIO(response.content), 'rt') as f:
        data_kdd = pd.read_csv(f, header=None)
else:
    raise ValueError("Download failed")

# Assign column names
data_kdd.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
                    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

# Group labels into 5 categories
attack_mapping = {
    'normal.': 0,
    'back.': 1, 'land.': 1, 'neptune.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1,
    'ipsweep.': 2, 'nmap.': 2, 'portsweep.': 2, 'satan.': 2,
    'ftp_write.': 3, 'guess_passwd.': 3, 'imap.': 3, 'multihop.': 3, 'phf.': 3, 'spy.': 3, 'warezclient.': 3, 'warezmaster.': 3,
    'buffer_overflow.': 4, 'loadmodule.': 4, 'perl.': 4, 'rootkit.': 4
}
data_kdd['label'] = data_kdd['label'].map(attack_mapping)
if data_kdd['label'].isnull().any():
    print("Unknown labels found. Setting to 0 (normal).")
    data_kdd['label'].fillna(0, inplace=True)

# Convert categorical columns to numerical codes
data_kdd['protocol_type'] = data_kdd['protocol_type'].astype('category').cat.codes
data_kdd['service'] = data_kdd['service'].astype('category').cat.codes
data_kdd['flag'] = data_kdd['flag'].astype('category').cat.codes

# Subsample
data_kdd = data_kdd.sample(n=10000, random_state=42)

# Simple feature selection using information gain
X_kdd = data_kdd.iloc[:, :-1]
y_kdd = data_kdd['label'].values
gains = {}
for feature in X_kdd.columns:
    values = X_kdd[feature].unique()
    best_gain = -np.inf
    for value in values:
        left_idx = X_kdd[feature] == value
        right_idx = ~left_idx
        left_y, right_y = y_kdd[left_idx], y_kdd[right_idx]
        parent_ent = entropy(y_kdd)
        weighted_ent = (len(left_y) / len(y_kdd)) * entropy(left_y) + (len(right_y) / len(y_kdd)) * entropy(right_y)
        gain = parent_ent - weighted_ent
        best_gain = max(best_gain, gain)
    gains[feature] = best_gain if best_gain != -np.inf else 0
selected_features = sorted(gains, key=gains.get, reverse=True)[:20]
print(f"Selected features: {selected_features}")
data_kdd_selected = data_kdd[selected_features + ['label']]
data_kdd_selected.to_csv('kdd_selected.csv', index=False)
print(f"KDD prepared: {data_kdd_selected.shape}\n{data_kdd_selected.head()}")

# Data Splitting (Train, Validation, Test)
def split_data(df, train_ratio=0.8, val_from_train=0.2, random_state=42):
    np.random.seed(random_state)
    groups = df.groupby('label')
    train_df = []
    test_df = []
    for _, group in groups:
        n = len(group)
        train_size_g = int(n * train_ratio)
        train_g = group.sample(n=train_size_g, random_state=random_state)
        test_g = group.drop(train_g.index)
        train_df.append(train_g)
        test_df.append(test_g)
    train_df = pd.concat(train_df)
    test_df = pd.concat(test_df)
    groups_train = train_df.groupby('label')
    val_df = []
    train_final = []
    for _, group in groups_train:
        n = len(group)
        val_size_g = int(n * val_from_train)
        val_g = group.sample(n=val_size_g, random_state=random_state)
        train_f = group.drop(val_g.index)
        val_df.append(val_g)
        train_final.append(train_f)
    val_df = pd.concat(val_df)
    train_df = pd.concat(train_final)
    return train_df, val_df, test_df

data_kdd = pd.read_csv('kdd_selected.csv')

# Handle missing values
if data_kdd.isnull().sum().any():
    for col in data_kdd.columns[:-1]:
        if data_kdd[col].dtype == 'float64':
            data_kdd[col].fillna(data_kdd[col].mean(), inplace=True)
        else:
            data_kdd[col].fillna(data_kdd[col].mode()[0], inplace=True)
# Split data
train_kdd, val_kdd, test_kdd = split_data(data_kdd)
train_kdd.to_csv('kdd_train.csv', index=False)
val_kdd.to_csv('kdd_val.csv', index=False)
test_kdd.to_csv('kdd_test.csv', index=False)
print(f"KDD: Train {len(train_kdd)}, Val {len(val_kdd)}, Test {len(test_kdd)}")

# Discretization
def discretize_quartile(df_train, df_other, features):
    for feature in features:
        if df_train[feature].dtype in ['float64', 'int64'] and df_train[feature].nunique() > 20:
            df_train[feature] = pd.qcut(df_train[feature], q=10, labels=False, duplicates='drop')
            bins = pd.qcut(df_train[feature], q=10, retbins=True, duplicates='drop')[1]
            df_other[feature] = pd.cut(df_other[feature], bins=bins, labels=range(len(bins)-1), include_lowest=True)
    return df_train, df_other

# Load split data and discretize
train_kdd = pd.read_csv('kdd_train.csv')
val_kdd = pd.read_csv('kdd_val.csv')
test_kdd = pd.read_csv('kdd_test.csv')
features_kdd = train_kdd.columns[:-1]
train_kdd, val_kdd = discretize_quartile(train_kdd, val_kdd, features_kdd)
train_kdd, test_kdd = discretize_quartile(train_kdd, test_kdd, features_kdd)
train_kdd.to_csv('kdd_train_discretized.csv', index=False)
val_kdd.to_csv('kdd_val_discretized.csv', index=False)
test_kdd.to_csv('kdd_test_discretized.csv', index=False)

# Build decision tree recursively
def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='entropy', class_weight=None):
    node = TreeNode(depth)
    if len(y) == 0:
        node.is_leaf = True
        node.label = 0
        node.gini = 0
        node.info_gain = 0
        return node
    node.label = Counter(y).most_common(1)[0][0] if len(y) > 0 else 0
    if len(y) < min_samples_split or depth >= max_depth or len(set(y)) == 1:
        node.is_leaf = True
        node.gini = gini(y, class_weight)
        node.info_gain = 0
        return node
    feature, value, gain, gini_val = best_split(X, y, criterion, class_weight)
    if feature is None:
        node.is_leaf = True
        node.gini = gini(y, class_weight)
        return node
    node.feature = feature
    node.value = value
    node.info_gain = gain
    node.gini = gini_val
    left_idx = X[feature] == value
    right_idx = ~left_idx
    left_y = y[left_idx]
    right_y = y[right_idx]
    if len(left_y) < min_samples_leaf or len(right_y) < min_samples_leaf:
        node.is_leaf = True
        node.gini = gini(y, class_weight)
        return node
    if len(left_y) > 0:
        node.children[value] = build_tree(X[left_idx], left_y, depth+1, max_depth, min_samples_split, min_samples_leaf, criterion, class_weight)
    else:
        child = TreeNode(depth=depth+1, is_leaf=True)
        child.label = node.label
        child.gini = 0
        node.children[value] = child
    if len(right_y) > 0:
        node.children['other'] = build_tree(X[right_idx], right_y, depth+1, max_depth, min_samples_split, min_samples_leaf, criterion, class_weight)
    else:
        child = TreeNode(depth=depth+1, is_leaf=True)
        child.label = node.label
        child.gini = 0
        node.children['other'] = child
    return node

# Predict for a single sample
def predict(node, sample):
    if node.is_leaf:
        return node.label
    child_key = sample[node.feature] if sample[node.feature] == node.value else 'other'
    if child_key in node.children:
        return predict(node.children[child_key], sample)
    return node.label

# Calculate accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Pruning with validation set
def prune_tree(node, X_val, y_val):
    if len(y_val) < 10:
        return 0
    if node.is_leaf:
        preds = [node.label] * len(y_val)
        errors = sum(p != t for p, t in zip(preds, y_val))
        return errors
    left_idx = X_val[node.feature] == node.value
    right_idx = ~left_idx
    left_child = node.children.get(node.value)
    right_child = node.children.get('other')
    left_errors = prune_tree(left_child, X_val[left_idx], y_val[left_idx]) if left_child else 0
    right_errors = prune_tree(right_child, X_val[right_idx], y_val[right_idx]) if right_child else 0
    errors_kept = left_errors + right_errors
    preds_leaf = [node.label] * len(y_val)
    errors_leaf = sum(p != t for p, t in zip(preds_leaf, y_val))
    if errors_leaf <= errors_kept:
        node.is_leaf = True
        node.children = {}
        return errors_leaf
    return errors_kept

# Visualize tree with graphviz
def visualize_tree(node, graph=None):
    if graph is None:
        graph = Digraph()
    label = f"Feature: {node.feature}\\nValue: {node.value}\\nInfo Gain: {node.info_gain:.4f}\\nGini: {node.gini:.4f}"
    if node.is_leaf:
        label = f"Leaf: Class {node.label}\\nGini: {node.gini:.4f}"
    graph.node(str(id(node)), label)
    for val, child in node.children.items():
        graph.edge(str(id(node)), str(id(child)), label=str(val))
        visualize_tree(child, graph)
    return graph

# Print tree structure
def print_tree(node, indent=""):
    if node.is_leaf:
        print(f"{indent}Leaf: Class {node.label}, Gini: {node.gini:.4f}")
    else:
        print(f"{indent}Node: Feature {node.feature}, Value {node.value}, Info Gain: {node.info_gain:.4f}, Gini: {node.gini:.4f}")
        for key, child in node.children.items():
            print(f"{indent}Branch {key}:")
            print_tree(child, indent + "  ")

# Train and Evaluate for KDD
train_kdd = pd.read_csv('kdd_train_discretized.csv')
X_train_kdd = train_kdd.iloc[:, :-1]
y_train_kdd = train_kdd.iloc[:, -1].values
val_kdd = pd.read_csv('kdd_val_discretized.csv')
X_val_kdd = val_kdd.iloc[:, :-1]
y_val_kdd = val_kdd.iloc[:, -1].values
test_kdd = pd.read_csv('kdd_test_discretized.csv')
X_test_kdd = test_kdd.iloc[:, :-1]
y_test_kdd = test_kdd.iloc[:, -1].values

# Hyperparameter tuning
best_acc = 0
best_params = {}
for crit in ['entropy', 'gini']:
    for max_d in [3, 5, 7, 10]:
        for min_s in [2, 5, 10]:
            for min_l in [2, 5, 10]:
                counts = Counter(y_train_kdd)
                n_classes = len(counts)
                class_weight = {cls: np.sqrt(len(y_train_kdd) / (n_classes * counts[cls])) for cls in counts if counts[cls] > 0}
                tree = build_tree(X_train_kdd, y_train_kdd, max_depth=max_d, min_samples_split=min_s, min_samples_leaf=min_l, criterion=crit, class_weight=class_weight)
                prune_tree(tree, X_val_kdd, y_val_kdd)
                preds_val = np.array([predict(tree, row) for _, row in X_val_kdd.iterrows()])
                acc = accuracy(y_val_kdd, preds_val)
                if acc > best_acc and acc > 0.5:
                    best_acc = acc
                    best_params = {'max_depth': max_d, 'min_samples_split': min_s, 'min_samples_leaf': min_l, 'criterion': crit}

# Combine training and validation sets for final model training
train_val_kdd = pd.concat([train_kdd, val_kdd], axis=0)

# Create a new validation set for pruning
val_from_train_val = 0.2
groups = train_val_kdd.groupby('label')
new_val = []
train_final = []
for _, group in groups:
    n = len(group)
    val_size_g = int(n * val_from_train_val)
    val_g = group.sample(n=val_size_g, random_state=42)
    train_g = group.drop(val_g.index)
    new_val.append(val_g)
    train_final.append(train_g)
new_val_kdd = pd.concat(new_val)
train_final_kdd = pd.concat(train_final)

# Prepare features and labels for final training and new validation
X_train_final_kdd = train_final_kdd.iloc[:, :-1]
y_train_final_kdd = train_final_kdd.iloc[:, -1].values
X_new_val_kdd = new_val_kdd.iloc[:, :-1]
y_new_val_kdd = new_val_kdd.iloc[:, -1].values

# Compute class_weight for final training
counts_final = Counter(y_train_final_kdd)
n_classes_final = len(counts_final)
class_weight_final = {cls: np.sqrt(len(y_train_final_kdd) / (n_classes_final * counts_final[cls])) for cls in counts_final if counts_final[cls] > 0}

# Train final tree
tree_kdd = build_tree(X_train_final_kdd, y_train_final_kdd, class_weight=class_weight_final, **best_params)
# Prune the final tree
prune_tree(tree_kdd, X_new_val_kdd, y_new_val_kdd)

# Evaluate on test set
preds_test_kdd = np.array([predict(tree_kdd, row) for _, row in X_test_kdd.iterrows()])
acc_test_kdd = accuracy(y_test_kdd, preds_test_kdd)
print(f"KDD Test Accuracy: {acc_test_kdd}")

# Display the final structure of the decision tree
print("\nFinal Decision Tree Structure:")
print_tree(tree_kdd)

# Display graph visualization
graph = visualize_tree(tree_kdd)
graph.render('kdd_decision_tree', format='pdf', view=True)
