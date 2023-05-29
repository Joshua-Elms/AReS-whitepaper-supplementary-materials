from utils import *
"""
Initialization Parameters:

        id (int) - request id for particular comparison run
        
        datapath (str) - a file path of the csv data

        which_regressors (dict) - dictionary of key:value pairs of form <'RegressorName'> : <Bool(0)|Bool(1)>

        metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized

        n_vizualized_tb (int) - the top scoring 'n' regressors over the test set to be included in final table. The value -1 will include all regressors (Default: -1)

        test_set_size (float) - a number between 0 and 1 that indicates the proportion of data to be allocated to the test set (Default: 0.2)

        n_cv_folds (int) - the number of folds for k-fold cross validation training (Default: 10)

        score_method (str) - the regressors will be evaluated on this metric to determine which regressors perform best (Default: 'Root Mean Squared Error')
                
        n_workers (int) - (n_workers != 1 ONLY PARTIALLY SUPPORTED) this determines whether the 'run' function is performed serially or with multiple concurrent processors. 
        
        figure_lst (list) - a list of the names of the figures to be generated on the frontend that require a separate process in the backend
"""

paramdict = {'id': 30,
            'datapath': 'datasets/test_datasets/concrete.csv',
            'which_regressors': {'ARDRegression': 1, 'AdaBoostRegressor': 1, 'BaggingRegressor': 1, 'BayesianRidge': 1, 'CCA': 0, 
                                 'DecisionTreeRegressor': 1, 'DummyRegressor': 0, 'ElasticNet': 1, 'ExtraTreeRegressor': 1, 
                                 'ExtraTreesRegressor': 1, 'GammaRegressor': 1, 'GaussianProcessRegressor': 0, 'GradientBoostingRegressor': 1, 
                                 'HistGradientBoostingRegressor': 1, 'HuberRegressor': 1, 'IsotonicRegression': 1, 'KNeighborsRegressor': 1, 
                                 'KernelRidge': 1, 'Lars': 1, 'Lasso': 1, 'LassoLars': 1, 'LassoLarsIC': 1, 'LinearRegression': 1, 
                                 'LinearSVR': 1, 'MLPRegressor': 0, 'MultiTaskElasticNet': 0, 'MultiTaskLasso': 0, 'NuSVR': 1, 
                                 'OrthogonalMatchingPursuit': 1, 'PLSCanonical': 0, 'PLSRegression': 0, 'PassiveAggressiveRegressor': 1, 
                                 'PoissonRegressor': 1, 'QuantileRegressor': 0, 'RANSACRegressor': 0, 'RadiusNeighborsRegressor': 1, 
                                 'RandomForestRegressor': 1, 'Ridge': 1, 'SGDRegressor': 1, 'SVR': 1, 'TheilSenRegressor': 0, 
                                 'TransformedTargetRegressor': 0, 'TweedieRegressor': 0
                                 },
            'metric_list': ['Mean Squared Error','Mean Absolute Error','R-Squared', 'Root Mean Squared Error'],
            'figure_lst': ['Test_Best_Models', 'Accuracy_over_Various_Proportions_of_Training_Set', 'Error_by_Datapoint'],
            'test_set_size': 0.2,
            'n_cv_folds': 10,
            'score_method': 'Root Mean Squared Error',
            'n_workers': 1,
            }


### Regular run ###
if __name__=="__main__":
    print(comparison_wrapper(2,paramdict)) # put 2 in the first arg to run in custom param mode, or 1 for defaults
