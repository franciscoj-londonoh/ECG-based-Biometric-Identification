import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score


def svmlinear_model(X, y, test_size=0.3, random_state=42):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split

    OUTPUT
    svmlinear - svm linear model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    y_preds - pandas dataframe, response variable 
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
                                                        random_state=random_state)
    
    # Predict the model
    svmlinear = svm.SVC(kernel='linear')
    svmlinear.fit(X_train, y_train)
    y_preds = svmlinear.predict(X_test)
    
    return X_train, X_test, y_train, y_test, svmlinear, y_preds



def model_performace(y_test, y_preds):
    '''
    INPUT
    y_test - pandas dataframe, X matrix
    y_preds - pandas dataframe, response variable
    
    OUTPUT
    model_accuracy - model's accuracy score
    model_f1score - model's f1 score
    model_recall - model's recall score
    model_precision - model's precision score
    rsquared_score - model's r-squared score
    '''
    
    model_accuracy = accuracy_score(y_test, y_preds)
    model_f1score = f1_score(y_test, y_preds, average='macro', labels=np.unique(y_preds))
    model_recall = recall_score(y_test, y_preds, average='macro', labels=np.unique(y_preds))
    model_precision = precision_score(y_test, y_preds, average='macro', labels=np.unique(y_preds))
    
    print("The accuracy for the model was {}.".format(model_accuracy))
    print("The f1 score for the model was {}.".format(model_f1score))
    print("The recall score for the model was {}.".format(model_recall))
    print("The precision score for the model was {}.".format(model_precision))
    
    rsquared_score = r2_score(y_test, y_preds)
    length_y_test = len(y_test)
    
    print("The r-squared score for your model was {} on {} values.".format(rsquared_score, length_y_test))

    return model_accuracy, model_f1score, model_recall, model_precision, rsquared_score


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - coefficients of the svm linear model 
    X_train - the training data, so the column names can be used
    
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    
    coefs_df = pd.DataFrame()
    coefs_df['feature'] = X_train.columns
    coefs_df['coefs'] = coefficients
    coefs_df['abs_coefs'] = np.abs(coefficients)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    
    return coefs_df


def find_optimal_model(X, y, cutoffs, test_size = .30, random_state=42, plot=True, cutoff_features = False):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars or number of features
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default True to plot result
    cutoff_features - boolean, default False to select cutoff of nonzero values or of number of features
    
    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    svmlinear - svm linear model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    best_cutoff - number of features that produces the model with the highest r-squared score
    '''
    
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        if cutoff_features == 0:
            reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        else:
            reduce_X = X.iloc[:, 0:cutoff]
        
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test, svmlinear, y_test_preds = svmlinear_model(reduce_X, y, test_size, random_state)
               
        y_train_preds = svmlinear.predict(X_train)
        
        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])
    
    X_train, X_test, y_train, y_test, svmlinear, y_test_preds = svmlinear_model(reduce_X, y, test_size, random_state)
        

    return r2_scores_test, r2_scores_train, svmlinear, X_train, X_test, y_train, y_test, best_cutoff