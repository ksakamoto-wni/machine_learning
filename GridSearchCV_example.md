

```python
# Grid Search Test
#
import numpy as np 
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import sys
import os.path
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def main():

#    args = sys.argv
#
#    if len(args) != 3:
#        print "usage: %s [Feature CSV] [Target CSV]" % (args[0])
#        sys.exit()
#
#    csv_feature = args[1]
#    csv_target = args[2]

    #-- Input Files
    csv_feature = "./probe_HANEDA.2017.txt"
    csv_target = "./probe_TATENO.2017.txt"

    files = [csv_feature, csv_target]
    for f in files:
        if not os.path.exists(f):
            print "%s is not found !" % f
            sys.exit()

    #-- Reading Feature
    data = pd.read_csv(csv_feature, delim_whitespace=True)
    X_all = data.loc[:, ['TEMP','U','V']]
    print "X_all= %d, %d" % (X_all.shape[0], X_all.shape[1])

    #-- Reading Target
    data2 = pd.read_csv(csv_target, delim_whitespace=True)
    Y_all = data2.loc[:,'TEMP'].as_matrix()
    print "Y_all= %d" % (Y_all.shape[0])

    #-- Model
    clf = GridSearchCV(
        MLPRegressor(hidden_layer_sizes=(100,),activation='relu',solver="adam",random_state=0,max_iter=20000),
        param_grid=[{'hidden_layer_sizes':[(4,),(10,),(40,),(100,),(200,),(300,),(400,),(500,),(4,4),(10,10),(100,100),(300,10)]} ],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=2,
        verbose=True
    )
    
    #-- Train the model with Grid Search
    clf.fit(X_all, Y_all)
        
    #-- Show all the parameters in Grid Search
    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)
   
    print ""
    print "Best estimator:"
    print clf.best_estimator_

    #-- Save the best model
    joblib.dump(clf.best_estimator_, './upper_NN_bestmodel')

    
    #-- Reference model
    clf_ref = MLPRegressor(hidden_layer_sizes=(10,),activation='relu',solver="adam",random_state=0,max_iter=20000)

    #-- Train the reference model
    clf_ref.fit(X_all, Y_all)

    #-- Save the reference model
    joblib.dump(clf_ref, './upper_NN_ref')


if __name__ == "__main__":
    main()

```

    X_all= 234, 3
    Y_all= 234
    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:   11.2s
    [Parallel(n_jobs=2)]: Done  60 out of  60 | elapsed:   12.8s finished
    /Users/ksakamoto/.pyenv/versions/anaconda2-4.4.0/lib/python2.7/site-packages/sklearn/model_selection/_search.py:667: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)


    -0.722 (+/- 0.326) for {'hidden_layer_sizes': (4,)}
    -0.586 (+/- 0.156) for {'hidden_layer_sizes': (10,)}
    -0.508 (+/- 0.149) for {'hidden_layer_sizes': (40,)}
    -0.407 (+/- 0.105) for {'hidden_layer_sizes': (100,)}
    -0.418 (+/- 0.167) for {'hidden_layer_sizes': (200,)}
    -0.372 (+/- 0.122) for {'hidden_layer_sizes': (300,)}
    -0.363 (+/- 0.109) for {'hidden_layer_sizes': (400,)}
    -2.419 (+/- 2.116) for {'hidden_layer_sizes': (500,)}
    -0.569 (+/- 0.270) for {'hidden_layer_sizes': (4, 4)}
    -0.685 (+/- 0.248) for {'hidden_layer_sizes': (10, 10)}
    -8.846 (+/- 2.619) for {'hidden_layer_sizes': (100, 100)}
    -6.134 (+/- 2.377) for {'hidden_layer_sizes': (300, 10)}
    
    Best estimator:
    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(400,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=20000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
           solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)

