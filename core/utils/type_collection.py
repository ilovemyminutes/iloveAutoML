from enum import Enum


class EstimatorTypes(Enum):
    DecisionTree = "dt"
    RandomForest = "rf"
    SVM = "svm"
    LogisticRegression = "lr"
    LinearRegreesion = "reg"


class ProblemTypes(Enum):
    MultiClass = "multiclass"
    Binary = "binary"
    regression = "regression"


class DataTypes(Enum):
    MarketingTrain = "tests/samples/marketing/marketing_train.csv"
    MarketingTest = "tests/samples/marketing/marketing_test.csv"

    Marketing_NaNs_Train = "tests/samples/marketing_NaNs/marketing_with_NaNs_train.csv"
    Marketing_NaNs_Test = "tests/samples/marketing_NaNs/marketing_with_NaNs_test.csv"

    LendingClubTrain = "tests/samples/lendingclub/lendingclub_train.csv"
    LendingClubTest = "tests/samples/lendingclub/lendingclub_test.csv"

    TitanicTrain = "tests/samples/titanic/titanic_train.csv"
    TitanicTest = "tests/samples/titanic/titanic_test.csv"