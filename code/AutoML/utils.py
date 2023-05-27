import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import all_estimators
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.base import clone
from sklearn.decomposition import PCA
import multiprocessing as multiprocessing

import warnings
warnings.filterwarnings('ignore')
from inspect import signature, _empty

def validation(datapath: str) -> None:
    """
    This functon validates if a dataset is of all numeric type and does not exceed the dimensions of 10,000 x 100.
    An error will be raised describing any of breech of these requirements by the dataset.

    Args:

        datapath (str) - a file path of a csv file

    Returns:

        will return True if the dataset fits requirements; will raise an exception otherwise
    """

    dataset, _, _ = load_data(datapath)
    issues_w_data = [*size(dataset, 10000, 100), *dtype_check(dataset)]
    for issue in issues_w_data:
        if issue:
            raise Exception(issue)
    pass


def size(dataset: pd.DataFrame, row_max: int, col_max: int) -> set:
    """
    This function will validate that a pandas dataframe is of dimensions less than the specified row and column maximums.

    Args:

        dataset (pd.Dataframe) - a pandas dataframe representing a dataset

        row_max (int) - the maximum number of rows allowed for a dataset

        col_max (int) - the maximum number of columns allowed for a dataset

    Returns:

        error (set) - a set of errors describing the way in which a dataframe exceeds the size limit
    """

    error = set()
    if dataset.shape[0] > row_max:
        error.add(f'The number of rows in the dataset exceeds {row_max}. Please reduce the number of rows.')
    if dataset.shape[1] > col_max:
        error.add(f'The number of columns in the dataset exceeds {col_max}. Please reduce the number of columns.')
    return error


def dtype_check(dataset: pd.DataFrame) -> set:
    """
    This function will validate the datatypes present in the dataframe. Allowable datatypes are int64 and float64. The presence
    of any other datatype in the dataframe will cause a descriptive error to be thrown.

    Args:

        dataset (pd.Dataframe) - a pandas dataframe representing a dataset

    Returns:

        error (set) - a set of errors describing the disallowed datatypes present in the dataset
    """

    dtypes = dataset.dtypes.to_dict()
    error = set()
    for col_name, typ in dtypes.items():
        if typ == 'bool':
            error.add('The dataset contains a boolean column. Please convert dataset to all numeric values.')
        if typ == 'O':
            error.add('This dataset contains a categorial data column, a date/time data column, or there has been data input error. Please convert dataset to all numeric values.')
    return error


def get_all_regs(which_regressors: dict) -> list:
    """
    This function imports all sklearn regression estimators. The function will filter all out all regressors
    that take additional parameters. It will return a list of all viable regressor classes and a list of the
    names of all the viable regressor classes.

    Args:

        which_regressors (dict) - dictionary of key:value pairs of form <'RegressorName'> : <Bool(0)|Bool(1)>

    Returns:

        regressors (lists) - two lists, the first being all regressor objects, the seconds being the corresponding regressor names
    """

    # importing all sklearn regressors and establishing which regressors will be ommited from the run
    estimators = all_estimators(type_filter='regressor')
    all_regs = []
    all_reg_names = []

    # removing regressors that require additional parameters or those that are cross-validation variants of existing regressors
    for name, RegressorClass in estimators:
        params = [val[1] for val in signature(RegressorClass).parameters.items()]
        all_optional = True
        for param in params:
            if param.default == _empty:
                all_optional = False
        is_cv_variant = name[-2:] == "CV"
        if all_optional and not is_cv_variant and which_regressors[name] == 1:
            reg = RegressorClass()
            all_regs.append(reg)
            all_reg_names.append(name)
    return all_regs, all_reg_names


def load_data(datapath: str) -> pd.DataFrame:
    """
    This function will take the relative file path of a csv file and return a pandas DataFrame of the csv content.

    Args:

        datapath (str) - a file path of the csv data

    Returns:

        raw data (pd.DataFrame) - a pandas dataframe containing the csv data
    """

    try:
        csv_path = os.path.abspath(datapath)
        df = pd.read_csv(csv_path)
        return df, df.iloc[:, :-1], df.iloc[:, -1]

    except Exception as e:
        raise ValueError(f"Expected a valid path to data - invalid: {csv_path}")


def data_split(datapath: str, test_set_size: float) -> tuple:
    """
    This function will take a relative datapath of a dataset in csv format and will split the data into training attributes,
    training labels, test attributes.

    Args:

        datapath (str) - a file path (eventually from s3 bucket) of the csv data

        test_set_size (float) - a number between 0 and 1 that indicates the proportion of data to be allocated to the test set (Default: 0.2)

    Returns:

        train/test datasets (tuple) - Four pandas dataframes: the first is training set attributes, the second is training set
                                      labels, the third is test set attributes, the fourth is test set labels
    """

    # the data is loaded
    _, attribs, labels = load_data(datapath)

    # the training and test sets are created
    split = ShuffleSplit(n_splits=1, test_size=test_set_size)
    for train_index, test_index in split.split(attribs, labels):
        train_attribs = attribs.loc[train_index]
        train_labels = labels.loc[train_index]
        test_attribs = attribs.loc[test_index]
        test_labels = labels.loc[test_index]

    return (train_attribs, train_labels, test_attribs, test_labels)


def gen_cv_samples(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, n_cv_folds: int = 11) -> tuple:
    """
    Generates a nested array of length k (where k is the number of cv folds).
    Each sub-tuple contains k folds formed into training data and the k+1 fold left out as test data.

    Args:

        X_train_df (pd.DataFrame) - training data already processed

        y_train (pd.DataFrame) - training labels already processed

        n_cv_folds (int) - the number of folds for k-fold cross validation training (Default: 10)

    Returns:

        train/test data (tuples) - nested_samples gets broken down into four lists
    """

    X_train, y_train = X_train_df.values, y_train_df.values
    kf = KFold(n_splits = n_cv_folds, shuffle = True) # KFold creates a generator object, not list
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)] # making list of indices to be used for folds based on KFold object
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices] # unpacking train/test data @ train/test indices 
    X_tr, y_tr, X_te, y_te = [], [], [], [] # variables which will each be of type list(np.ndarray, np.ndarray,..., np.ndarray), with k ndarray's representing each fold
    for sample in nested_samples:
        for i, var in enumerate((X_tr, y_tr, X_te, y_te)):
            var.append(sample[i]) # method to prevent code duplication in unpacking nested_samples into four variables
    return (X_tr, y_tr, X_te, y_te)


def metric_help_func():
    """
    Internal table to assist with any functions involving metrics

    Args:

        None

    Returns:

        metric_table (dict) - dictionary of general form: { 'metric': [ higher score is better?, positive or negative score values, accociated stat function ] } 
    """

    def root_mean_squared_error(y_true, y_pred, multioutput="uniform_average"):
        return metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput=multioutput)**(1/2)

    metric_table = {'Explained Variance': {'Correlation Score': True, 'Function': metrics.explained_variance_score, 'Multi-Output': True},
                    'Max Error': {'Correlation Score': False, 'Function': metrics.max_error, 'Multi-Output': False},
                    'Mean Absolute Error': {'Correlation Score': False, 'Function': metrics.mean_absolute_error, 'Multi-Output': True},
                    'Mean Squared Error': {'Correlation Score': False, 'Function': metrics.mean_squared_error, 'Multi-Output': True},
                    'Root Mean Squared Error': {'Correlation Score': False, 'Function': root_mean_squared_error, 'Multi-Output': True},
                    # 'Mean Squared Log Error': {'Correlation Score': False, 'Function': metrics.mean_squared_log_error, 'Multi-Output': True},
                    'Median Absolute Error': {'Correlation Score': False, 'Function': metrics.median_absolute_error, 'Multi-Output': True},
                    'R-Squared': {'Correlation Score': True, 'Function': metrics.r2_score, 'Multi-Output': True},
                    'Mean Poisson Deviance': {'Correlation Score': False, 'Function': metrics.mean_poisson_deviance, 'Multi-Output': False},
                    'Mean Gamma Deviance': {'Correlation Score': False, 'Function': metrics.mean_gamma_deviance, 'Multi-Output': False},
                    'Mean Absolute Percentage Error': {'Correlation Score': False, 'Function': metrics.mean_absolute_percentage_error, 'Multi-Output': True},
                    'D-Squared Absolute Error Score': {'Correlation Score': True, 'Function': metrics.d2_absolute_error_score, 'Multi-Output': True},
                    'D-Squared Pinball Score': {'Correlation Score': True, 'Function': metrics.d2_pinball_score, 'Multi-Output': True},
                    'D-Squared Tweedie Score': {'Correlation Score': True, 'Function': metrics.d2_tweedie_score, 'Multi-Output': False}
                   }
    
    try:
        return metric_table

    except Exception as e:
        raise Exception("Update your version of sklearn to comply with requirements.txt")


def preprocess(train_attribs: np.array, train_labels: np.array, test_attribs: np.array, test_labels: np.array) -> tuple:
    """
    This function will standardize data attributes and impute NaN values via KNN-Imputation for the entire dataset.

    Args:

        train_attribs (np.ndarray) - np.ndarray of training attributes

        train_labels (np.ndarray) - np.ndarray of training labels

        test_attribs (np.ndarray) - np.ndarray of test attributes

        test_labels (np.ndarray) - np.ndarray of test labels

    Returns:

        train_attribs_prepped (np.ndarray) - np.ndarray of training attributes that have been standardized and had NaN values imputed

        train_labels_prepped (np.ndarray) - np.ndarray of training labels that have had NaN values imputed

        test_attribs_prepped (np.ndarray) - np.ndarray of test attributes that have been standardized and had NaN values imputed

        test_labels_prepped (np.ndarray) - np.ndarray of test labels that have had NaN values imputed
    """

    # standardizing attributes
    scaler = StandardScaler()
    scaler.fit(train_attribs)
    train_attribs = scaler.transform(train_attribs)
    test_attribs = scaler.transform(test_attribs)

    # joining attributes and labels in order to perform KNN-Imputation
    full_train = np.concatenate((train_attribs, np.expand_dims(train_labels, axis=1)), axis=1)
    full_test = np.concatenate((test_attribs, np.expand_dims(test_labels, axis=1)), axis=1)
    imputer = KNNImputer()
    imputer.fit(full_train)
    imp_full_train = imputer.transform(full_train)
    imp_full_test = imputer.transform(full_test)

    # splitting attributes from labels once again
    train_attribs_prepped = imp_full_train[:, :-1]
    train_labels_prepped = imp_full_train[:, -1]
    test_attribs_prepped = imp_full_test[:, :-1]
    test_labels_prepped = imp_full_test[:, -1]

    return (train_attribs_prepped, train_labels_prepped, test_attribs_prepped, test_labels_prepped)


def comparison_wrapper(setting: int, conf: dict) -> dict:
    """
    This function is a wrapper for the comparison function. Based on the setting of the wrapper, the comparison function will either be run with
    default parameters or with specified parameters.

    Args:

        setting (int): 1 to indicate a request from the basic user interface and 2 to indicate a request from the advanced user interface

        conf (dict): A dictionary of hyperparameters to be sent to the run function. If the dictionary contains only an id and datapath, the other 
                    hyperparameters will be imputed with default values.
    """

    default_conf = {'id': conf['id'],
            'which_regressors': {'ARDRegression': 1, 'AdaBoostRegressor': 1, 'BaggingRegressor': 1, 'BayesianRidge': 1, 'CCA': 0, 
                                 'DecisionTreeRegressor': 1, 'DummyRegressor': 0, 'ElasticNet': 1, 'ExtraTreeRegressor': 1, 
                                 'ExtraTreesRegressor': 1, 'GammaRegressor': 1, 'GaussianProcessRegressor': 0, 'GradientBoostingRegressor': 1, 
                                 'HistGradientBoostingRegressor': 1, 'HuberRegressor': 1, 'IsotonicRegression': 0, 'KNeighborsRegressor': 1, 
                                 'KernelRidge': 0, 'Lars': 1, 'Lasso': 1, 'LassoLars': 1, 'LassoLarsIC': 1, 'LinearRegression': 1, 
                                 'LinearSVR': 1, 'MLPRegressor': 0, 'MultiTaskElasticNet': 0, 'MultiTaskLasso': 0, 'NuSVR': 1, 
                                 'OrthogonalMatchingPursuit': 1, 'PLSCanonical': 0, 'PLSRegression': 1, 'PassiveAggressiveRegressor': 1, 
                                 'PoissonRegressor': 1, 'QuantileRegressor': 0, 'RANSACRegressor': 1, 'RadiusNeighborsRegressor': 1, 
                                 'RandomForestRegressor': 1, 'Ridge': 1, 'SGDRegressor': 0, 'SVR': 1, 'TheilSenRegressor': 0, 
                                 'TransformedTargetRegressor': 1, 'TweedieRegressor': 1
                                 }, 
            'metric_list': ['Explained Variance', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 
                            'Median Absolute Error', 'R-Squared', 'Mean Poisson Deviance', 'Mean Gamma Deviance', 
                            'Mean Absolute Percentage Error', 'D-Squared Absolute Error Score',
                            'D-Squared Pinball Score', 'D-Squared Tweedie Score'], 
            'n_vizualized_tb': -1, 
            'test_set_size': 0.1,
            'n_cv_folds': 11, 
            'score_method': 'Root Mean Squared Error',
            'datapath': conf['datapath'], 
            'n_workers': 1,
            'figure_lst': ['Accuracy_over_Various_Proportions_of_Training_Set', 'Error_by_Datapoint', 'Test_Best_Models'] # 'Accuracy_over_Various_Proportions_of_Training_Set', 'Error_by_Datapoint', 'Test_Best_Models'
                }
    if setting == 1:
        return comparison(**default_conf)
    elif setting == 2:
        return comparison(**conf)
    else:
        raise Exception("The setting for the comparison function must be either 1 (to indicate request from basic user interface) or 2 (to indicate request from advanced user interface)")



def comparison(id: int, which_regressors: dict, metric_list: list, n_vizualized_tb: int, test_set_size: float,
               n_cv_folds: int, score_method: str, datapath: str, n_workers: int, figure_lst: list) -> list:
    """
    This function will perform cross-validation training across several regressor types for one dataset. It will
    also deploy other functions to generate additional visualizations.

    Args:

        id (int) - request id for particular comparison run
        
        datapath (str) - a file path (eventually from s3 bucket) of the csv data

        which_regressors (dict) - dictionary of key:value pairs of form <'RegressorName'> : <Bool(0)|Bool(1)>

        metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized

        n_vizualized_tb (int) - the top scoring 'n' regressors over the test set to be included in final table. The value -1 will include all regressors (Default: -1)

        test_set_size (float) - a number between 0 and 1 that indicates the proportion of data to be allocated to the test set (Default: 0.2)

        n_cv_folds (int) - the number of folds for k-fold cross validation training (Default: 10)

        score_method (str) - the regressors will be evaluated on this metric to determine which regressors perform best (Default: 'Root Mean Squared Error')
        
        datapath (str) - a file path to temporary dataset file retrieved from input s3 bucket
        
        n_workers (int) - this determines whether the 'run' function is performed serially or with multiple concurrent processors. The user selects the number of processes (Default: 1)
        
        figure_lst (list) - a list of the names of the figures to be generated on the frontend that require a separate process in the backend

    Returns:
        
        failed_regs (list) - a list of regressors that encountered an error in cross-validation training
    """

    # validating dataset
    validation(datapath)

    regs, reg_names = get_all_regs(which_regressors)
    train_attribs, train_labels, test_attribs, test_labels = data_split(datapath, test_set_size)
    train_attribs_idx, train_labels_idx, test_attribs_idx, test_labels_idx = list(train_attribs.index), list(train_labels.index), list(test_attribs.index), list(test_labels.index)

    # appending the score method to the metric list to be used in the remainder of the program
    metric_list = [score_method] + metric_list
    for i, item in enumerate(metric_list[1:]):
        if item == metric_list[0]:
            del metric_list[i + 1]

    metric_help = metric_help_func()

    # creating cv samples and running each regressor over these samples
    cv_X_train, cv_y_train, cv_X_test, cv_y_test = gen_cv_samples(train_attribs, train_labels, n_cv_folds)
    # fundemental idea of args_lst is to create the cross product of all k folds with all r regressors, making k*r tasks (sets of arguments) to be passed to mp pool
    # to do this, below list comp will use two diff indices - [i // n_cv_folds] to group all regressors of same type and [i % n_cv_folds] to split those regressors over each of the k (normally 10) folds
    # could be done just as well with a nested for loop iterating over both regressors and folds
    args_lst = [(regs[i // n_cv_folds], reg_names[i // n_cv_folds], metric_list, metric_help, cv_X_train[i % n_cv_folds], cv_y_train[i % n_cv_folds],
                 cv_X_test[i % n_cv_folds], cv_y_test[i % n_cv_folds]) for i in range(len(regs) * n_cv_folds)]

    if n_workers == 1:  # serial
        results = [run(*args) for args in args_lst]

    else:  # parallel
        multiprocessing.set_start_method("spawn")  # spawn method is safer and supported across both Unix and Windows systems, alternative (may not work) is fork
        with multiprocessing.Pool(processes=n_workers) as pool:  # defaulting to 8 processesors
            results = pool.starmap(run, args_lst)

    # organizing results of cv runs into a dictionary and collecting regressors that failed cross-validation
    failed_regs = set()
    org_results = {}  # -> {'Reg Name': [{'Same Reg Name': [metric, metric, ..., Reg Obj.]}, {}, {}, ... ], '':[], '':[], ... } of raw results
    for success_status, single_reg_output in results:
        if success_status:
            reg_name = list(single_reg_output.keys())[0]
            if reg_name in org_results:
                org_results[reg_name] += [single_reg_output]
            else:
                org_results[reg_name] = [single_reg_output]

        else:
            failed_regs.add(single_reg_output)

    # keeping only those results that did not throw an error during any cv run
    fin_org_results = {k: v for k, v in org_results.items() if k not in failed_regs}
    assert fin_org_results, f"All regressors failed"

    path_gen = lambda file: f"perf_stats_{file}_{id}.csv" # helper for making various output data files
    
    
    # the figure lookup dict has to include the parameters that will be passed to any functions it calls
    figure_lookup = {'Accuracy_over_Various_Proportions_of_Training_Set': (gen_and_write_training_test_data, (
                        regs, reg_names, train_attribs, train_labels, path_gen('Accuracy_over_Various_Proportions_of_Training_Set'), metric_list, metric_help
                        )),
                    'Error_by_Datapoint': (error_viz, (
                        fin_org_results, train_attribs, train_labels, test_attribs, test_labels, 
                        train_attribs_idx, train_labels_idx, test_attribs_idx, test_labels_idx, n_cv_folds, metric_help, path_gen('Error_by_Datapoint')
                        )),
                    'Test_Best_Models': (test_best, (
                        fin_org_results, metric_list, train_attribs, train_labels, test_attribs, test_labels, metric_help, n_vizualized_tb, 
                        path_gen('Test_Best_Models')
                        ))
                    }
    
    for fig, (func, params) in figure_lookup.items():
        if fig in figure_lst:
            func(*params) # these functions will both create AND write out the data - they do not return anything

    output_path = f"perf_stats_{id}.csv"
    write_results(output_path, fin_org_results, metric_list)

    return {
        'output_path': output_path,
        'failed_regs': list(failed_regs)
        }


def run(reg: object, reg_name: str, metric_list: list, metric_help: dict, train_attribs: np.ndarray, train_labels: np.ndarray,
        test_attribs: np.ndarray, test_labels: np.ndarray) -> dict:
    """
    This function will perform cross-validation training on a given dataset and given regressor. It will return
    a dictionary containing cross-validation performance on various metrics.

    Args:

        reg (object) - a scikit-learn regressor object

        reg_name (str) - the associated scikit-learn regressor name

        metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized

        metric_help (dict) - a dictionary to assist with any functions involving metrics

        train_attribs (np.ndarray) - np.ndarray of training attributes

        train_labels (np.ndarray) - np.ndarray of training labels

        test_attribs (np.ndarray) - np.ndarray of test attributes

        test_labels (np.ndarray) - np.ndarray of test labels

    Returns:

        reg_dict (dict) - dictionary of results from cross-validation run on one regressor
    """
    success = True
    try:
        # preprocessing data
        train_attribs, train_labels, test_attribs, test_labels = preprocess(train_attribs, train_labels, test_attribs, test_labels)
        clone_reg = clone(reg)
        model_trained = clone_reg.fit(train_attribs, train_labels)
        y_pred = model_trained.predict(test_attribs)
        reg_dict = {reg_name: []}
        for k in metric_list:
            calculated = metric_help[k]['Function'](test_labels, y_pred)
            reg_dict[reg_name].append(calculated)
        reg_dict[reg_name].append(model_trained)

    except Exception as e:
        success = False
        reg_dict = reg_name

    return success, reg_dict



def test_best(fin_org_results: dict, metric_list: list, train_attribs: np.array, train_labels: np.array, test_attribs: np.array, test_labels: np.array, metric_help: dict, 
              n_vizualized_tb: int, path: str):
    """
    This function will take the best performing model on each regressor type generated by cross-validation training and 
    apply it to the set of test data. The performance of the regs on the test instances will be written to a csv file.
    The regressors will be sorted in descending order by performance on specified metrics.

    Args:

    fin_org_results (dict) - the final results from cross-validation

    metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized

    train_attribs (np.array) - a numpy array of training set attributes

    test_attribs (np.array) - a numpy array of test set attributes

    train_labels (np.array) - a numpy array of training set labels

    test_labels (np.array) - a numpy array of test set labels

    metric_help (dict) - a dictionary to assist with any functions involving metrics

    n_vizualized_tb (int) - the top scoring 'n' regressors over the test set to be included in final table. The value -1 will include all regressors (Default: -1)

    path (str) - the path to write final CSV results to
    
    Returns:

        A csv displaying the top performing model of each regressor type. The "best" models are determined by using the highest scoring model on cross-validation
        and using it to predict the labels of the test set. The models will be listed best-to-worst by their prediction performance on the test set.
    """

    columns = metric_list
    rows = []
    output = []

    # loops over each regressor type
    for k, v in fin_org_results.items():
        rows.append(k)

        # storing each of the 'k' scores for each model over 'k' cross-validation runs. the metric used to determine best score is specified by the user.
        # also stores the corresponding sci-kit learn regressor object
        scores = [list(dict.values())[0][0] for dict in v]
        models = [list(dict.values())[0][-1] for dict in v]

        # if the specified score metric is a loss metric, the model with the lowest score will be "best". if the specified metric is a correlation score
        # (like R^2), then the model with the highest score will be "best"
        if metric_help[metric_list[0]]['Correlation Score'] == True:
            best = max(zip(scores, models), key = lambda pair: pair[0])[1]
        else:
            best = min(zip(scores, models), key = lambda pair: pair[0])[1]

        # preprocessing data
        train_attribs, train_labels, test_attribs, test_labels = preprocess(train_attribs, train_labels, test_attribs, test_labels)
        # using the "best" model to predict the test labels
        best_predict = best.predict(test_attribs)

        # calculating the difference between predictions and ground-truth labels
        single_reg_output = []
        for m in metric_list:
            calculated = metric_help[m]['Function'](test_labels, best_predict)
            single_reg_output.append(round(calculated, 4))

        output.append(single_reg_output)

    # creating a table to display the prediction score of the "best" model of each regressor type. the regressors are ranked according to the best performance over
    # test label predictions
    df = pd.DataFrame(data=output, index=rows, columns=columns)

    df_sorted = df.sort_values(by=columns[0], axis=0, ascending=not (metric_help[columns[0]]['Correlation Score']))
    df_sorted = df_sorted.iloc[:n_vizualized_tb]
    df_sorted.to_csv(path, header=True, index=True)

    return


def write_results(path: str, data: dict, metrics: list) -> None:
    """
    An internal function to create a write a csv file from the data of a dictionary of a specific format

    Args:

        path (str) - the path of the file to be written

        data (dict) - the dictionary to be converted to csv

        metrics (list) - the regressors will be evaluated on these metrics during cross-validation and visualized

    Returns:

        None
    """

    acc = {f"{regr}~{metric}": [] for regr in data for metric in metrics}
    for regressor, runs in data.items():
        for fold, run in enumerate(runs):
            for metric_idx, value in enumerate(list(run.values())[0]):
                if metric_idx < len(metrics):
                    acc[f"{regressor}~{metrics[metric_idx]}"].append(value)

    df = pd.DataFrame(acc)
    df.to_csv(path)


def error_viz(fin_org_results: dict, train_attribs: pd.DataFrame, train_labels: pd.DataFrame, test_attribs: pd.DataFrame, test_labels: pd.DataFrame, 
              train_attribs_idx: list, train_labels_idx: list, test_attribs_idx: list, test_labels_idx: list, n_cv_folds: int, metric_help: dict, 
              path: str, metrics_presented="Mean Absolute Percentage Error"):
    """
    This function generates a CSV file that stores various error metrics for the prediction of each point for each regressor

    Args:

        fin_org_results (dict) - the final results from cross-validation

        train_attribs (pd.DataFrame) - pd.DataFrame of dataset training attributes

        train_labels (pd.DataFrame) - pd.DataFrame of dataset training labels

        test_attribs (pd.DataFrame) - pd.DataFrame of dataset test attributes

        test_labels (pd.DataFrame) - pd.DataFrame of dataset test labels

        train_attribs_idx (list) - list of the indexes of the shuffled training set

        train_labels_idx (list) - list of the indexes of the shuffled training set

        test_attribs_idx (list) - list of the indexes of the shuffled test set

        test_labels_idx (list) - list of the indexes of the shuffled test set

        n_cv_folds (int) - the number of folds for k-fold cross validation training

        metric_help (dict) - internal dictionary to assist with metrics

        path (str) - the path to write final CSV results to

        metrics_presented (str) - the metrics to show in final csv: can be either "Mean Absolute Percentage Error" or "All" (Default: "Mean Absolute Percentage Error")

    Returns:

        writes a CSV file to specified path
    """

    train_attribs, train_labels, test_attribs, test_labels = preprocess(train_attribs, train_labels, test_attribs, test_labels)
    X = pd.DataFrame(np.vstack((train_attribs, test_attribs)), index=train_attribs_idx+test_attribs_idx)
    y = pd.DataFrame(np.vstack((np.expand_dims(train_labels, axis=1), np.expand_dims(test_labels, axis=1))), index=train_labels_idx+test_labels_idx)
    y = y.values.reshape((len(y),))

    all_dfs = []
    for reg_name, res in fin_org_results.items():
        partial_row = [[] for _ in range(n_cv_folds)]
        print(f'Predicting point by point with {reg_name}')
        for fold_idx, fold in enumerate(res):
            if reg_name == "PLSRegression":
                y_pred = list(fold[reg_name][-1].predict(X).flatten().tolist())
            else:
                y_pred = list(fold[reg_name][-1].predict(X))
            partial_row[fold_idx] += y_pred
        partial_row.append(y)

        point_data = pd.DataFrame(partial_row, columns=[f'{reg_name}~p{idx}' for idx in train_labels_idx+test_labels_idx])
        y_true = [list(np.array(point_data.tail(1)).squeeze()) for _ in range(n_cv_folds)]
        y_pred = [point_data.loc[i, :].values.flatten().tolist() for i in range(n_cv_folds)]

    # All metrics configuration
        if metrics_presented == "All":
            
            score = np.array([v['Function'](y_true=y_true, y_pred=y_pred, multioutput="raw_values") for k,v in metric_help.items() if v['Multi-Output']]) 
            metric_scores = pd.DataFrame(score, index=[k for k,v in metric_help.items() if v['Multi-Output']==True], columns=point_data.columns)
            metric_scores = metric_scores.transpose()

            for column in metric_scores.columns:
                if metric_help[column]['Correlation Score']:
                    metric_scores[column] = -1* metric_scores[column]

            df_standard = metric_scores.copy(deep=True)

            for column in df_standard.columns:
                if column == 'Mean Absolute Percentage Error':
                    df_standard['Raw Mean Absolute Percentage Error'] = df_standard[column].copy(deep=True)
                    df_standard[column] = (df_standard[column] - df_standard[column].mean()) / df_standard[column].std()
                elif column in ['D-Squared Absolute Error Score','D-Squared Pinball Score']:
                    df_standard.drop(column, axis=1)
                else:
                    df_standard[column] = (df_standard[column] - df_standard[column].mean()) / df_standard[column].std()

            error_ensemble = error_PCA(df_standard.copy(deep=True))
            df_standard['Error Ensemble'] = error_ensemble.values
            df_standard['Error Ensemble'] = (df_standard['Error Ensemble'] - df_standard['Error Ensemble'].mean()) / df_standard['Error Ensemble'].std()


    # Mean Absolute Percentage Error configuration
        elif metrics_presented == "Mean Absolute Percentage Error":

            score = np.array([metric_help[metrics_presented]['Function'](y_true=y_true, y_pred=y_pred, multioutput="raw_values")]) 
            metric_scores = pd.DataFrame(score, index=["Raw Mean Absolute Percentage Error"], columns=point_data.columns)
            metric_scores = metric_scores.transpose()

            df_standard = metric_scores.copy(deep=True)

        else:
            raise Exception("metrics_presented must be either 'All' or 'Mean Absolute Percentage Error'")

        all_dfs.append(df_standard)

    final_df = pd.concat(all_dfs, ignore_index=False)
    final_df.to_csv(path, header=True, index=True)
    return


def error_PCA(error_data: pd.DataFrame, verbose=False) -> pd.Series:
    """
    This function fits a PCA model on a dataset and then reduces that dataset to a singular dimension. The function may also print
    several attributes of the PCA model.

    Args:
        
        error_data (pd.DataFrame) - a standardized DataFrame of various error metrics on each point for each regressor
        
        verbose (bool) - will print information about PCA model if True

    Returns:

        error_pca (pd.DataFrame) - a one-column DataFrame of the error_data projected to one dimension using PCA

    """
    pca = PCA(n_components=1)
    error_data = error_data.to_numpy()
    print(error_data.shape)
    pca.fit(error_data)
    error_pca = pd.DataFrame(pca.transform(error_data))

    if verbose:
        print(f"""
        Components: {pca.components_}
        Explained Variance: {pca.explained_variance_}
        Explained Variance Ratio: {pca.explained_variance_ratio_}
        Singular Values: {pca.singular_values_}
        Mean: {pca.mean_}
        Number of Components: {pca.n_components_}
        Number of Features: {pca.n_features_}
        Number of Samples: {pca.n_samples_}
        Number of Features seen: {pca.n_features_in_}
        """)

    return error_pca


def gen_and_write_training_test_data(regs, reg_names, X, y, path: str, metric_list: list, metric_help: dict):
    """
    There are many iterations occurring here. Each of the following items is looped over:
        1. fold_sets - each of these is a different group of 10/11 folds that will potentially be used to train [((trainfold1, ... trainfoldn), testfold), ..., ((trainfold1, ... trainfoldn), testfold)]
        2. pcnt_folds - from 1-10, how many folds should be used to train the model before testing on the 11th fold. Corresponds to 10-100% of training data used
        3. regs and reg_names - iterable of all regressors and their names to be tested
    
    Args:

    Returns:
     
        None, but writes a similar output to the main output for this program - differences are that this
        output is indexed by percentage of training data, not cv fold - metrics computed over the cv folds are averaged here.
        Also, this includes both train and test data, so there should be twice as many columns compared to the normal output of just test accuracy.
    """
    ITERS = 10
    # ITERS = 10  
    FOLDS = 11
    # FOLDS = 10
    cv_X_train, cv_y_train, cv_X_test, cv_y_test = gen_cv_samples(X, y, FOLDS)
    pcnts = np.arange(10, 110, 10)
    ## generate fold_sets
    gen_offsetted = lambda data, offset: [data[(i+offset) % len(data)] for i in range(len(data))]
    train_outputs = []
    test_outputs = []
    for i in range(FOLDS):
        # X_train, X_test, y_train, y_test = cv_X_train[i], cv_X_test[i], cv_y_train[i], cv_y_test[i]
        X_train, X_test, y_train, y_test = [gen_offsetted(data, i) for data in (cv_X_train[0], cv_X_test[0], cv_y_train[0], cv_y_test[0])]
        for n_folds in range(1, ITERS+1):
            # X_tr = np.vstack(X_train[:n_folds])
            # X_te = np.vstack(X_test[:n_folds])
            # y_tr = np.hstack(y_train[:n_folds])
            # y_te = np.hstack(y_test[:n_folds])
            size = cv_X_test[0].shape[0]
            X_tr = np.vstack(X_train[:n_folds*size])
            X_te = np.vstack(X_test[:n_folds*size])
            y_tr = np.hstack(y_train[:n_folds*size])
            y_te = np.hstack(y_test[:n_folds*size])
            for reg, reg_name in zip(regs, reg_names):
                train_output = run(reg, reg_name, metric_list, metric_help, X_tr, y_tr.flatten(), X_tr, y_tr.flatten()) # remove models from end of data
                test_output = run(reg, reg_name, metric_list, metric_help, X_tr, y_tr.flatten(), X_te, y_te.flatten())
                
                train_outputs.append((i, n_folds-1, *train_output))
                test_outputs.append((i, n_folds-1, *test_output))
   
                
    
    # processing round 1 - extract useful data from output of all runs
    fin_org_results_d = {}
    failed_regs = set()
    for tt_name, tt_out in (("train", train_outputs), ("test", test_outputs)):
        org_results = {name: [[None for _ in range(ITERS)] for _ in range(FOLDS)] for name in reg_names} # -> {'Reg Name': [{'Same Reg Name': [metric, metric, ..., Reg Obj.]}, {}, {}, ... ], '':[], '':[], ... } of raw results
        for iter, amt, success_status, single_reg_output in tt_out:
            if success_status:
                reg_name = list(single_reg_output.keys())[0]
                org_results[reg_name][iter][amt] = single_reg_output[reg_name][:-1]
                    
            else:
                failed_regs.add(single_reg_output)
                
        fin_org_results_d[tt_name] = {k: np.array(v).mean(axis=0).tolist() for k,v in org_results.items() if k not in failed_regs}
                
                
    json = {tt: {reg_name: {metric: {pcnt: [] for pcnt in pcnts} for metric in metric_list} for reg_name in reg_names if reg_name not in failed_regs} for tt in ("train", "test")}

    # processing round 2 - use previous representation of data to get data into a clean JSON format
    for tt_name in ("train", "test"):
        for regressor, averages in fin_org_results_d[tt_name].items():
            if regressor not in failed_regs:
                for n_folds, values in enumerate(averages):
                    for metric_idx, value in enumerate(values):
                        if metric_idx < len(metric_list):
                            
                            json[tt_name][regressor][metric_list[metric_idx]][(((n_folds % (ITERS)) + 1) * 10)].append(value)

    # reshape data to work in a csv format (pd.dataframe)
    output_dict = {f"{regr}~{metric}~{tt}": [] for regr in reg_names if regr not in failed_regs for metric in metric_list for tt in ("train", "test")}
    for tt_name in ("train", "test"):
        for reg_name in reg_names:
            if reg_name not in failed_regs:
                for metric in metric_list:
                    for pcnt in pcnts:
                        value = json[tt_name][reg_name][metric][pcnt][0]
                        output_dict[f"{reg_name}~{metric}~{tt_name}"].append(value)
                        

    df = pd.DataFrame(output_dict)
    df["percent_training_data"] = pcnts
    df.set_index("percent_training_data", inplace = True)
    df.to_csv(path, header=True, index=True)
        
    return 