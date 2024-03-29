# 🤖iloveAutoML
* 본 저장소는 머신러닝 파이프라인 개발 과정을 기록하기 위한 공간입니다.
* 조금씩 기능을 확장해나갈 예정입니다😊

## Samples
* data/target: marketing/insurance_subscribe, titanic/Survived
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
![](https://github.com/iloveslowfood/iloveAutoML/blob/main/results/Train%20Phase.png?raw=true)
```python
>>> python main.py --input samples/marketing/marketing_train.csv --mode train --problem_type binary --target insurance_subscribe --save_path results
Train report has been saved in 'results/dt_YYYYMMDDHHMMSS'.
```
* **Test Phase**
![](https://github.com/iloveslowfood/iloveAutoML/blob/main/results/Test%20Phase.png?raw=true)
```python
>>> python main.py --input samples/marketing/marketing_test.csv --mode test --meta_path results/dt_YYYYMMDDHHMMSS --save_path results
Test report has been saved in 'results/prediction_dt_YYYYMMDDHHMMSS(marketing_test.csv)'.
```

## Output
```
results                                                         # input path to save results
  ├─dt_YYYYMMDDHHMMSS                                           # output folder of train phase
  │    ├─dt_YYYYMMDDHHMMSS.zip                                  # zip of meta files
  │    └─dt_report_YYYYMMDDHHMMSS.xlsx                          # train report
  └─prediction_dt_YYYYMMDDHHMMSS(marketing_test.csv)            # output folder of test phase
       └─prediction_dt_YYYYMMDDHHMMSS(marketing_test.csv).xlsx  # test report
```

* **Train Report Sample**
    * Report: [dt_report_20201027163041.xlsx](https://github.com/iloveslowfood/iloveAutoML/raw/main/tests/results/dt_20201027163041/dt_report_20201027163041.xlsx)
    * Meta file: [dt_20201027163041.zip](https://github.com/iloveslowfood/iloveAutoML/raw/main/tests/results/dt_20201027163041/dt_20201027163041.zip)

* **Test Report Sample**
    * Report: [prediction_dt_20201027163041(marketing_test.csv).xlsx](https://github.com/iloveslowfood/iloveAutoML/raw/main/results/prediction_dt_20201028174759(marketing_with_NaNs_test.csv)/prediction_dt_20201028174759(marketing_with_NaNs_test.csv).xlsx)
    
## TODO
* More algorithms
* More transformations
* Improved tuning process
