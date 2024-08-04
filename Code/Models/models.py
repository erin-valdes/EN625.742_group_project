from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.utils.validation import column_or_1d

class Model():
    def __init__(self, Train, Test, X, y, model=LogisticRegression()):
        '''
        Generic Class for a model which we will use to classify

        inputs:
            Train:  Training data, already split before being passed here
            Test:   Test data, already split before being passed her
            X:      an array of explanatory variables ['e', 'H', 'moid_jup' ]
            y:      an array of the response variable for example ['pha']

        FIXME:  If i've already fit a model and THEN i cross validate the same
        instance, does it barf? Does that affect fit at all? maybe add one more field for the model spec, separate from the fit model?
        '''
        self.model  = model # Model to train and predict with
        self.Train  = Train # Train Dataset
        self.Test   = Test  # Test Dataset
        self.X      = X     # Explanatory columns
        self.y      = y     # Response Columns
        self.y_hat  = None  # Predicted Values
        self.prepData()
    
    def prepData(self):
        '''
        Take the explanatory and response variables and filter Train/Test to only
        contain those columns
        '''
        # Concatenate Explanatory and response variables
        fields = np.concatenate( [self.X, self.y] )
        # Filter data
        self.Train  = self.Train[ fields ]
        self.Test   = self.Test[ fields ]
    
    def fit(self):
        '''
        Fit the instance's model with input data
        '''
        X = self.Train[self.X]
        y = column_or_1d(self.Train[self.y])
        self.model = self.model.fit(X, column_or_1d(y))
    
    def cross_validate(self, k=10):
        '''
        Run k-fold cross validation for the model

        input:
            k: the number of folds, default 10
        '''
        X = self.Train[self.X]
        y = column_or_1d(self.Train[self.y])
        scores = cross_val_score(self.model, X, y, cv=k)
        return( [np.mean(scores), np.std(scores) ] )
    
    def predict(self):
        '''
        Take the fitted model, split the test set into X and y, and
        predict values based on our fitted model.
        '''
        X = self.Test[ self.X ]
        self.y_hat = self.model.predict()
    
    def accuracy(self):
        '''
        Look at our classified y values, compare them to 
        '''
        x = np.absolute(self.y_hat - self.y) # since the array should be binary, any differences in predictions will show up as 1 or -1
        return( sum(x) / len(x) )


if __name__ == '__main__':
    pass