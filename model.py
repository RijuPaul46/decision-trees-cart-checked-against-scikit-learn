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
            left_gini=gini(left)
            right_gini=gini(right)
            score=float(parent_score-((len(left)*left_gini) +len(right)*right_gini)/n_sample)
            if score>gain:
                best_threshold=threshold
                best_feature=j
                gain=score
    return (best_feature,best_threshold,gain)

# Step 3 - grow_tree (not yet solved)
# TODO: implement

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

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

