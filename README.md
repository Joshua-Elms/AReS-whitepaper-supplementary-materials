# AReS Startup Guide

 <img src=".assets/ares_logo.png" alt= 'ares' width=100> <img src=".assets/hoosiers.png" alt= “IU” width=100 height=128>

This repository is intended to allow readers of "AReS: An AutoML Regression Service for Data Analytics and Novel Data-centric Visualizations" \[[paper]()\] to replicate the authors' results.

The [website](https://dalkilic.luddy.indiana.edu/) referred to in the paper runs code which is functionally identical to the AutoML code in this repository.

## Setup
To set up for a run of this pipeline, you will need to download and install the necessary libraries. Please ensure you have a python >= 3.7. Determine whether you are using pip or conda (if you don't know, use the instructions for pip). 

Pip users should run:
```
pip install -r requirements.txt
```

Conda users should run:
```
conda create --name <env> --file environment.yaml
conda activate <env>
```

## Testing
To run AReS for yourself, you will have to set the parameters for the run in `code/AutoML/main.py`. This file calls `code/AutoML/utils.py`, which contains all of the computation. `main.py` is already configured for a sample run, but some of the notable parameters that can be set are

* `datapath`

    The filepath of the data that you would like to run AReS on. AReS will attempt to predict the last column using the rest of the columns. The case studies presented in the paper are located at `datasets/kaggle_datasets` ([concrete](https://www.kaggle.com/competitions/playground-series-s3e9) and [paris](https://www.kaggle.com/competitions/playground-series-s3e6)), but other examples can be found in `datasets/test_datasets`.

* `which_regressors`
    
    This dictionary determines which of `sklearn`'s regressors will be used by AReS, indicated with `1`s.

* `figure_lst`

    This is a list of the figures discussed in the paper will have their data produced. We do not provide code to duplicate our visualizations here, as the paper shows images from a web-based tool called [Apache ECharts](https://echarts.apache.org/en/index.html).

See the rest of the parameters in `code/AutoML/main.py`

Once the parameters have been set, run AReS with
```
python code/AutoML/main.py
```

Output will be produced in your working directory.

## Analysis
The code for AReS produces up to four files, which are detailed below
1. `perf_stats_<id>.csv`
  
    Columns are `<regressor>~<error_metric>` for each regressor and error metric specified in config parameters. Rows are the 1-n folds specified for k-fold cross validation. Each value indicates how well that model (1 of n for each regressor) performed in cross validation according to a particular metric.

2. `perf_stats_quantity_curve_<id>.csv`

    Columns are `<regressor>~<error_metric>` for each regressor and error metric specified. Rows are 10-100, indicating what percent of the available data models in that row were trained on.

3. `perf_stats_point_errors_<id>.csv`

    Column 0 is `<regressor>~p<point_num>`. Column 1 is Mean Absolute Percentage Error (MAPE). Each value in Column 1 indicates how well that regressor performed over the particular point, according to its MAPE score.

4. `perf_stats_test_best_models_<id>.csv`

    Column 0 is the particular regressor. Columns 1 and on are metrics. Each value indicates the how well the regressor performed in prediction on the initially held-out data, according to some error metric. Only the m best models are shown here, according to the parameter set in program configuration.

All of the figures in the paper can be generated using the above generated data. AReS performance against other Kaggle submissions can be determined by
1. Set AReS' `datapath` parameter to point to a dataset downloaded from a [Kaggle competition](https://www.kaggle.com/competitions). 
2. Run AReS to predict the target values for the downloaded dataset.
3. Find the best RMSE from `perf_stats_test_best_models_<id>.csv` for use in step 6.
4. Download the submission leaderboard for that competition ([example](https://www.kaggle.com/competitions/playground-series-s3e9/leaderboard)) and record the file name (not path) for step 6. 
5. Place leaderboard file under `datasets/kaggle_leaderboards/`.
6. Run each cell in `code/case_studies/create_distribution.ipynb` after setting `ARES_RMSE` and `LEADERBOARD` from steps 3 and 4, respectively.

Email Joshua Elms (joshua.elms111@gmail.com) for questions.

If you find this work useful, cite it using:
```
@article{elms2023ares,
  title={AReS: An AutoML Regression Service for Data Analytics and Novel Data-centric Visualizations},
  author={Elms, Josh and Johnson, Sam and Kalkunte Ramachandra, Madhavan and Sugasi, Keerthana and Sharma, Parichit, and Kurban, Hasan and Dalkilic, Mehmet M.},
  year={2023}
}
```
