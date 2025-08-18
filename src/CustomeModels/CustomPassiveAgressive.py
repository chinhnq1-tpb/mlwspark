from sklearn.linear_model import PassiveAggressiveClassifier

class CustomPassiveAggressiveClassifier(PassiveAggressiveClassifier):
    def __init__(self, loss='hinge', *, C=1.0, max_iter=1000, random_state=None, tol=1e-3, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, fit_intercept=True, class_weight=None, warm_start=False, average=False):
        """
        Inherits from PassiveAggressiveClassifier and adds the predict_prob method.
        
        loss: {'hinge', 'log'}, default='hinge'
            The loss function to be used. If 'log' is specified, it allows probability prediction.
            
        C: float, default=1.0
            Regularization strength. Smaller values specify stronger regularization.
        
        max_iter: int, default=1000
            The maximum number of iterations.
        
        random_state: int, RandomState instance, default=None
            The seed for random number generation.
        
        Other parameters are passed to the base PassiveAggressiveClassifier.
        """
        super().__init__(loss=loss, C=C, max_iter=max_iter, random_state=random_state, tol=tol, early_stopping=early_stopping,
                         validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, fit_intercept=fit_intercept,
                         class_weight=class_weight, warm_start=warm_start, average=average)
    
    def predict_proba(self, X):
        """
        Method to predict probabilities for each class.
        Uses the _predict_proba_lr method from Logistic Regression.
        """
        # Compute probabilities using the internal method _predict_proba_lr
        return self._predict_proba_lr(X)