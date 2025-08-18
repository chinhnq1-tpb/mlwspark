from sklearn.metrics import roc_auc_score

def gini_score(y_true, y_scores):
    """
    Calculate the Gini coefficient based on AUC.

    Parameters:
    ----------
    y_true : array-like
        True binary labels (0 and 1).
    y_scores : array-like
        Target scores or probabilities estimated by a classifier.

    Returns:
    -------
    gini : float
        Gini coefficient (between -1 and 1).
    """
    auc = roc_auc_score(y_true, y_scores)
    return 2 * auc - 1
