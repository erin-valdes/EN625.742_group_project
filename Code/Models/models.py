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
        instance, does it barf? Does that affect fit at all?
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
        print(fields)
        # Filter data
        self.Train  = self.Train[ fields ]
        self.Test   = self.Test[ fields ]
    
    def fit(self):
        '''
        Fit the instance's model with input data
        '''
        X = self.Train[self.X]
        y = self.Train[self.y]
        self.model = self.model.fit(X, column_or_1d(y))
    
    def cross_validate(self, k=10):
        '''
        Run k-fold cross validation for the model

        input:
            k: the number of folds, default 10
        '''
        X = self.Train[self.X]
        y = self.Train[self.y]
        scores = cross_val_score(self.model, X, y, cv=k)
    
    def predict(self):
        '''
        Take the fitted model, split the test set into X and y, and
        predict values based on our fitted model.
        '''
        X = self.Test[ self.X ]
        self.y_hat = self.model.predict()



def Logistic(X, y):
    '''
    Takes in the explanatory (X) and response variables (y).
    runs 10 fold CV for a Logistic Regression Model, and then 
    trains a Logistic Regression Model.

    Returns the array of cross validation scores, and a model in an array
    [avg score, std dev,  model]
    '''
    # Fit the model
    y = column_or_1d(y)
    model = LogisticRegression().fit(X, y)

    # 10-fold CV
    scores  = cross_val_score(LogisticRegression(), X, y, cv=10)
    avg     = np.mean(scores)
    std     = np.std(scores)

    return( [avg, std, model] )

def LDA(X, y):
    '''
    Takes in the explanatory (X) and response variables (y).
    runs 10 fold CV for an LDA Model, and then trains a  Model. 
    '''
    # Fit the model
    model   = LinearDiscriminantAnalysis().fit(X, y)

    # 10-fold CV
    scores  = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=10)
    avg     = np.mean(scores)
    std     = np.std(scores)

    return( [avg, std, model] )

def QDA(X, y):
    '''
    Takes in the explanatory (X) and response variables (y).
    runs 10 fold CV for a QDA Model, and then trains a Model.
    '''
    # Fit the model
    model   = QuadraticDiscriminantAnalysis().fit(X, y)

    # 10-fold CV
    scores  = cross_val_score(QuadraticDiscriminantAnalysis(), X, y, cv=10)
    avg     = np.mean(scores)
    std     = np.std(scores)

    return( [avg, std, model] )

def SVM( X, y, kernel='linear'):
    '''
    Takes in the explanatory (X), response variables (y), and kernel.
    runs 10 fold CV for an SVM Model, and then trains a Model.

    Inputs:
        X:      explanatory vars
        y:      response vars
        kernel: Kernel to use for the analysis
    '''
    # Fit the model
    clf     = SVC( kernel=kernel )
    model   = SVC( kernel=kernel ).fit(X, y) 

    # 10-fold CV
    scores  = cross_val_score(clf, X, y, cv=10)
    avg     = np.mean(scores)
    std     = np.std(scores)
    return( [avg, std, model] )
    

if __name__ == '__main__':
    pass