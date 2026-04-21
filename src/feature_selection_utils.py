from sklearn.feature_selection import SelectKBest, mutual_info_classif

def apply_kbest_feature_selection(X_train, y_train, X_test, k=50):
    """
    Fit feature selection on training data only, then transform train and test sets.
    Returns selected train/test sets and the fitted selector.
    """
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector