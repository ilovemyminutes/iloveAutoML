# MiniCore

an orientation project of constructing Auto ML pipeline

## Dataset
* train data: [marketing_train.csv](https://gitlab.solidware.io/Andrew/minicore/blob/master/samples/marketing/marketing_train.csv)
* test data: [marketing_test.csv](https://gitlab.solidware.io/Andrew/minicore/blob/master/samples/marketing/marketing_test.csv)
* target: `insurance_subscribe`
* problem type: binary
* metrics: Accuracy, Precision, Recall, F1

## Run
* **Arguments**  
`--input`: path of input data, default="samples/marketing/marketing_train.csv"  
`--target`: name of target, default="insurance_subscribe"  
`--problem_type`: problem type, default="binary"  
`--estimator`: name of estimator needed for train, default="dt"  
`--mode`: one of 'train' and 'test', default="train"  
`--save_path`: path to save, default="results"  
`--cv`: k for cross validation, default=3  
`--meta_path`: path of mata data needed for test  

* **Train Phase**
```python
>>> python main.py --input samples/marketing/marketing_train.csv --mode train --problem_type binary --target insurance_subscribe --save_path results
Train report has been saved in 'results/dt_20201013185608'.
```
* **Test Phase**
```python
>>> python main.py --input samples/marketing/marketing_test.csv --mode test --meta_path results/dt_20201013185608 --save_path results
Test report has been saved in 'results/prediction_dt_20201013185608(marketing_test.csv)'.
```

## Output
```
results                                                         # input path to save results
  ├─dt_20201013185608                                           # output folder of train phase
  │    ├─dt_20201013185608.zip                                  # zip of meta files
  │    └─dt_report_20201013185608.xlsx                          # train report
  └─prediction_dt_20201013185608(marketing_test.csv)            # output folder of test phase
       └─prediction_dt_20201013185608(marketing_test.csv).xlsx  # test report
```

* **Train Report Sample**
    * Report: [dt_report_20201013185608.xlsx](https://gitlab.solidware.io/Andrew/minicore/raw/master/results/dt_20201013185608/dt_report_20201013185608.xlsx)
    * Meta file: [dt_20201013185608.zip](https://gitlab.solidware.io/Andrew/minicore/raw/master/results/dt_20201013185608/dt_20201013185608.zip)

* **Test Report Sample**
    * Report: [prediction_dt_20201013185608(marketing_test.csv).xlsx](https://gitlab.solidware.io/Andrew/minicore/raw/master/results/prediction_dt_20201013185608(marketing_test.csv)/prediction_dt_20201013185608(marketing_test.csv).xlsx)

## References
* [ML Core](https://gitlab.solidware.io/DataLabs/MLCore), Solidware
* [Julia's OT Repo](https://gitlab.solidware.io/julia_lee/ot-task1-julia), Solidware
* [AutoKeras](https://github.com/keras-team/autokeras), AutoKeras
* [Digest](https://gitlab.solidware.io/Andrew/digest), Solidware