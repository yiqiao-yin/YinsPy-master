class YinsML:

    """
    Yin's Machine Learning Package 
    Copyright © YINS CAPITAL, 2009 – Present
    """

    # Define function
    def DecisionTree_Classifier(X_train, X_test, y_train, y_test, maxdepth = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        
        # Train
        DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = DCT.predict(X_train)
        y_train_hat_score = DCT.predict_proba(X_train)

        from sklearn.metrics import confusion_matrix
        confusion_train = pd.DataFrame(confusion_matrix(y_train_hat_, y_train))
        confusion_train
        
        train_acc = sum(np.diag(confusion_train)) / sum(sum(np.array(confusion_train)))
        train_acc

        y_test_hat_ = DCT.predict(X_test)
        y_test_hat_score = DCT.predict_proba(X_test)
        confusion_test = pd.DataFrame(confusion_matrix(y_test_hat_, y_test))
        confusion_test

        test_acc = sum(np.diag(confusion_test)) / sum(sum(np.array(confusion_test)))
        test_acc
        
        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': DCT,
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'y_train_hat_score': y_train_hat_score,
                'confusion_train': confusion_train,
                'train_acc': train_acc
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'y_test_hat_score': y_test_hat_score,
                'confusion_test': confusion_test,
                'test_acc': test_acc
            }
        }
    # End of function
    
    # Define function
    def DecisionTree_Regressor(X_train, X_test, y_train, y_test, maxdepth = 3):
        
        # Import Modules
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import random
        from sklearn import tree
        
        # Train
        DCT = tree.DecisionTreeClassifier(max_depth=maxdepth)
        DCT = DCT.fit(X_train, y_train)
        
        # Report In-sample Estimators
        y_train_hat_ = DCT.predict(X_train)
        RMSE_train = np.sqrt(np.mean((y_train_hat_ - y_train)**2))

        # Report Out-of-sample Estimators
        y_test_hat_ = DCT.predict(X_test)
        RMSE_test = np.sqrt(np.mean((y_test_hat_ - y_test)**2))
        
        # Output
        return {
            'Data': {
                'X_train': X_train, 
                'y_train': y_train, 
                'X_test': X_test, 
                'y_test': y_test
            },
            'Model': DCT,
            'Train Result': {
                'y_train_hat_': y_train_hat_,
                'RMSE_train': RMSE_train
            },
            'Test Result': {
                'y_test_hat_': y_test_hat_,
                'RMSE_test': RMSE_test
            }
        }
    # End of function
    
    # Define function
    def ResultAUCROC(y_test, y_test_hat):
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
        areaUnderROC = auc(fpr, tpr)
        resultsROC = {
            'false positive rate': fpr,
            'true positive rate': tpr,
            'thresholds': thresholds,
            'auc': round(areaUnderROC, 3) }

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: \
                  Area under the curve = {0:0.3f}'.format(areaUnderROC))
        plt.legend(loc="lower right")
        plt.show()