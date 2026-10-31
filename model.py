"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
def gini(y):
    # TODO: 1 - sum(p_k^2); 0.0 for an empty array.
    freq={}
    n=len(y)
    if n==0:
        return 0.0
    for x in y:
        freq[x]=freq.get(x,0)+1
    gni=1
    for u,v in freq.items():
        gni-=((v/n)**2)
    return gni
    pass

# Step 2 - best_split
MIN_SAMPLE_LEAF=1
def best_split(X, y):
    # TODO: exhaustive search over features and midpoint thresholds; return (feature, threshold, gain).
    parent_score=gini(y)
    best_threshold=None
    best_feature=None
    gain=0.0
    n_sample,n_feature=X.shape
    for j in range(n_feature):
        col=sorted(set(X[:,j]))
        for i in range(1,len(col)):
            threshold=float((col[i]+col[i-1])/2)
            left=y[X[:,j]<=threshold]
            right=y[X[:,j]>threshold]
            if len(left)<MIN_SAMPLE_LEAF or len(right)<MIN_SAMPLE_LEAF:
                continue
            left_gini=gini(left)
            right_gini=gini(right)
            score=float(parent_score-((len(left)*left_gini) +len(right)*right_gini)/n_sample)
            if score>gain:
                best_threshold=threshold
                best_feature=j
                gain=score
    return (best_feature,best_threshold,gain)

# Step 3 - grow_tree
def majority_class(y):
    classes,counts=np.unique(y,return_counts=True)
    return classes[np.argmax(counts)]
def helper_best_split(X,y,MIN_SAMPLE_LEAF=1):
    parent_score=gini(y)
    best_threshold=None
    best_feature=None
    gain=0.0
    n_sample,n_feature=X.shape
    for j in range(n_feature):
        col=sorted(set(X[:,j]))
        for i in range(1,len(col)):
            threshold=float((col[i]+col[i-1])/2)
            left=y[X[:,j]<=threshold]
            right=y[X[:,j]>threshold]
            if len(left)<MIN_SAMPLE_LEAF or len(right)<MIN_SAMPLE_LEAF:
                continue
            left_gini=gini(left)
            right_gini=gini(right)
            score=float(parent_score-((len(left)*left_gini) +len(right)*right_gini)/n_sample)
            if score>gain:
                best_threshold=threshold
                best_feature=j
                gain=score
    return (best_feature,best_threshold,gain)
def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    # TODO: recursive CART with nested dicts; leaves hold the majority class.
    majority=int(majority_class(y))
    gini_score=gini(y)
    
    #pure class no need to split more 
    if gini_score==0:
        
        
        tree={
            "leaf":True,
            "value":majority,
            "n":len(y)
        }
        return tree
    #max depth reached
    if depth>=max_depth:
        return {
            "leaf":True,
            "value":majority,
            "n":len(y)
        }
    

    best_feature,best_threshold,gain=helper_best_split(X,y,min_samples_leaf)

    if best_feature is None or gain<=0:
        tree={
            "leaf":True,
            "value":majority,
            "n":len(y)
        }
        return tree
    #select all row where X[best_feature]<=threshold
    row_mask_left=X[:,best_feature]<=best_threshold
    row_mask_right=X[:,best_feature]>best_threshold
    left_X=X[row_mask_left]
    left_y=y[row_mask_left]
    right_X=X[row_mask_right]
    right_y=y[row_mask_right]

    tree={
        "leaf":False,
        "feature":best_feature,
        "threshold":best_threshold,
        "n": len(X),
        "left":grow_tree(left_X,left_y,max_depth,min_samples_leaf,depth+1),
        "right":grow_tree(right_X,right_y,max_depth,min_samples_leaf,depth+1)
    }
    return tree
    pass

# Step 4 - predict_tree
def predict_for_row(tree,x):
    if tree['leaf']:
        return tree['value']
    feature=tree['feature']
    threshold=tree['threshold']
    if x[feature]<=threshold:
        return predict_for_row(tree['left'],x)
    return int(predict_for_row(tree['right'],x))
    
def predict_tree(tree, X):
    # TODO: walk the dict tree per row; return an int array of leaf values.
    predictions=np.zeros(len(X),dtype=int)
    for i,x in enumerate(X):
        predictions[i]=int((predict_for_row(tree,x)))
    return predictions
    pass

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier
def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    # TODO: fit and return DecisionTreeClassifier(max_depth, min_samples_leaf, random_state).
    model=DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X,y)
    return model
    pass

# Step 6 - sklearn_splits
#by default sklearn save tree in depth first left first manner
#in clf.tree_ there is children_left,children_right,
#feature , threshold   tree.node_count save how many nodes are there
def sklearn_splits(clf):
    # TODO: [(feature, threshold)] for internal nodes in node-index order, thresholds rounded to 3 decimals.
    tree=clf.tree_
    result=[]
    for i in range(tree.node_count):
        if tree.children_left[i]==-1:
            continue
        threshold=round(float(tree.threshold[i]),2)
        feature=int(tree.feature[i])
        result.append((feature,threshold))
    return result

    pass

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

