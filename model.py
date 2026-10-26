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

# Step 2 - best_split (not yet solved)
# TODO: implement

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

