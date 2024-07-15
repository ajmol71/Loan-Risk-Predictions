import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import math
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

infile = "C:\\Users\\ajmol\\Documents\\Programming\\ML\\HMEQ_Loss_Imputed.csv"

df = pd.read_csv(infile)
df = df.drop("Unnamed: 0", axis = 1)

TargetL = "TARGET_LOSS_AMT"
TargetB = "TARGET_BAD_FLAG"

dt = df.dtypes

# SPLIT DATA

x = df.copy()
x = x.drop(TargetL, axis = 1)
x = x.drop(TargetB, axis = 1)
y = df[[TargetB, TargetL]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8, test_size = .2, random_state=1)

F = ~ y_train[TargetL].isna()
w_train = x_train[F]
z_train = y_train[F]

F = ~ y_test[TargetL].isna()
w_test = x_test[F]
z_test = y_test[F]

ROCList = []

def getProbAccuracyScores( NAME, MODEL, X, Y ) :
    pred = MODEL.predict( X )
    probs = MODEL.predict_proba( X )
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve( Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

def getAmtAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt(metrics.mean_squared_error(Y, pred))
    return [NAME, RMSE, MEAN]

rmsesall = []

## A2 Models
dectree = tree.DecisionTreeClassifier(max_depth = 4)
dectree = dectree.fit(x_train, y_train[TargetB])

ROCTree = getProbAccuracyScores("DecTree", dectree, x_test, y_test[TargetB])
ROCList.append(ROCTree)

# GET PREDICTIVE VARIABLES
def getTreeVars( TREE, varNames ) : # Thanks for the code
   tree_ = TREE.tree_
   varName = [ varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]

   nameSet = set()
   for i in tree_.feature :
       if i != _tree.TREE_UNDEFINED :
           nameSet.add( i )
   nameList = list( nameSet )
   parameter_list = list()
   for i in nameList :
       parameter_list.append( varNames[i] )
   return parameter_list

feature_cols = list(x.columns.values)
varsTreeFlag = getTreeVars(dectree, feature_cols)

# DAMAGES REGRESSION TREE
amtTree = tree.DecisionTreeRegressor(max_depth =3)
amtTree = amtTree.fit(w_train, z_train[TargetL])

zPredTrain = amtTree.predict(w_train)
zPredTest = amtTree.predict(w_test)

rmseTree = getAmtAccuracyScores("Tree", amtTree, w_test, z_test[TargetL])
rmsesall.append(rmseTree)

feature_cols = list(x.columns.values)
varsTreeAmt = getTreeVars(amtTree, feature_cols)

def getEnsembleTreeVars( ENSTREE, varNames ) :
   importance = ENSTREE.feature_importances_
   index = np.argsort(importance)
   theList = []
   for i in index :
       imp_val = importance[i]
       if imp_val > np.average( ENSTREE.feature_importances_ ) :
           v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
           theList.append( ( varNames[i], v ))
   theList = sorted(theList,key=itemgetter(1),reverse=True)
   return theList
# #
# # ##  RANDOM FOREST  ##
#
dectreeRF = RandomForestClassifier(n_estimators = 100, random_state=1)
dectreeRF = dectreeRF.fit(x_train, y_train[TargetB])

ROCRF = getProbAccuracyScores("RF", dectreeRF, x_test, y_test[TargetB])
ROCList.append(ROCRF)

feature_cols = list(x.columns.values)
varsRFFlag1 = getEnsembleTreeVars(dectreeRF, feature_cols)
varsRFFlag = []

for i in varsRFFlag1:
    varsRFFlag.append(i[0])
#
# RF Regressor to Predict Damages
RegRF = RandomForestRegressor(n_estimators = 100, random_state=1)
RegRF = RegRF.fit(w_train, z_train[TargetL])

Z_Pred_train = RegRF.predict(w_train)
Z_Pred_test = RegRF.predict(w_test)

rmseRF = getAmtAccuracyScores("RegRF", RegRF, w_test, z_test[TargetL])
rmsesall.append(rmseRF)

feature_cols = list(x.columns.values )
varsRFAmt1 = getEnsembleTreeVars(RegRF, feature_cols)

varsRFAmt = []
for i in varsRFAmt1:
    varsRFAmt.append(i[0])

# GRADIENT BOOSTING #

dectreeGB = GradientBoostingClassifier( random_state=1 )
dectreeGB = dectreeGB.fit(x_train, y_train[TargetB])

ROCGB = getProbAccuracyScores("GB", dectreeGB, x_test, y_test[TargetB])
ROCList.append(ROCGB)

feature_cols = list(x.columns.values)
varsGBFlag1 = getEnsembleTreeVars(dectreeGB, feature_cols)

varsGBFlag = []
for i in varsGBFlag1:
    varsGBFlag.append(i[0])

# GB Regression

RegGB = GradientBoostingRegressor(random_state=1)
RegGB = RegGB.fit(w_train, z_train[TargetL])

Z_Pred_train = RegGB.predict(w_train)
Z_Pred_test = RegGB.predict(w_test)

rmseGB = getAmtAccuracyScores("RegGB", RegGB, w_test, z_test[TargetL])
rmsesall.append(rmseGB)

feature_cols = list(x.columns.values )
varsGBAmt1 = getEnsembleTreeVars(RegGB, feature_cols)

varsGBAmt = []
for i in varsGBAmt1:
    varsGBAmt.append(i[0])


### A3 CODE ###

# Logistic Regression All Variables
m = LogisticRegression()
m.fit(x_train, y_train[TargetB])

varNames = list(x_train.columns.values)
coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varNames):
    coefdict[feat] = coef

ROCLogAll = getProbAccuracyScores("LogAll", m, x_test, y_test[TargetB])
ROCList.append(ROCLogAll)

# Logistic - Decision Tree Classifier
m_train = x_train[varsTreeFlag]
m_test = x_test[varsTreeFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsTreeFlag):
    coefdict[feat] = coef

ROCLogTree = getProbAccuracyScores("RF", m, m_test, y_test[TargetB])
ROCList.append(ROCLogTree)
#
# Logistic - RF Classifier
m_train = x_train[varsRFFlag]
m_test = x_test[varsRFFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsRFFlag):
    coefdict[feat] = coef

ROCLogRF = getProbAccuracyScores("LogRF", m, m_test, y_test[TargetB])
ROCList.append(ROCLogRF)

# Logistic - GB Classifier
m_train = x_train[varsGBFlag]
m_test = x_test[varsGBFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsGBFlag):
    coefdict[feat] = coef

ROCLogGB = getProbAccuracyScores("LogGB", m, m_test, y_test[TargetB])
ROCList.append(ROCLogGB)

# Logistic - Forward Selection
varNames = list(x_train.columns.values)
maxCols = x_train.shape[1]

sfs = SFS(LogisticRegression(),
          k_features = (1, maxCols),
          forward = True,
          floating = False,
          scoring = "r2",
          cv = 5)
sfs.fit(x_train.values, y_train[TargetB].values)

dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
dfmnames = dfm.columns.values

dfm = dfm[["feature_names", "avg_score"]]

dfm.avg_score = dfm.avg_score.astype(float)
maxIndex = dfm.avg_score.argmax()
theVars = dfm.iloc[maxIndex,]
theVars = theVars.feature_names

varsFSFlag = []
for i in theVars:
    index = int(i)
    try:
        theName = varNames[index]
        varsFSFlag.append(theName)
    except:
        pass


m_train = x_train[varsFSFlag]
m_test = x_test[varsFSFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsFSFlag):
    coefdict[feat] = coef

ROCLogFS = getProbAccuracyScores("LogFS", m, m_test, y_test[TargetB])
ROCList.append(ROCLogFS)


### LINEAR REGRESSION MODELS ###

print("\n\n\n")
m = LinearRegression()
m.fit(w_train, z_train[TargetL])

rmseLinTest = getAmtAccuracyScores("LinAll", m, w_train, z_train[TargetL])
rmsesall.append(rmseLinTest)


# Linear - Dec Tree Vars
m_train = w_train[varsTreeAmt]
m_test = w_test[varsTreeAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

rmseLinTree = getAmtAccuracyScores("LinTree", m, m_test, z_test[TargetL])
rmsesall.append(rmseLinTree)


# Linear - RF Vars
m_train = w_train[varsRFAmt]
m_test = w_test[varsRFAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

rmseLinRF = getAmtAccuracyScores("LinRF", m, m_test, z_test[TargetL])
rmsesall.append(rmseLinRF)


# Linear - GB Vars
m_train = w_train[varsGBAmt]
m_test = w_test[varsGBAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

rmseLinGB = getAmtAccuracyScores("LinGB", m, m_test, z_test[TargetL])
rmsesall.append(rmseLinGB)



# Linear - Forward Selection
varNames = list(w_train.columns.values)
maxCols = w_train.shape[1]

sfs = SFS(LinearRegression(),
          k_features = (1, maxCols),
          forward = True,
          floating = False,
          scoring = "r2",
          cv = 5)
sfs.fit(w_train.values, z_train[TargetL].values)

dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
dfmnames = dfm.columns.values

dfm = dfm[["feature_names", "avg_score"]]

dfm.avg_score = dfm.avg_score.astype(float)
maxIndex = dfm.avg_score.argmax()
theVars = dfm.iloc[maxIndex,]
theVars = theVars.feature_names

varsFSAmt = []
for i in theVars:
    index = int(i)
    try:
        theName = varNames[index]
        varsFSAmt.append(theName)
    except:
        pass


m_train = w_train[varsFSAmt]
m_test = w_test[varsFSAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

rmseLinFS = getAmtAccuracyScores("LinFS", m, m_test, z_test[TargetL])
rmsesall.append(rmseLinFS)


##### START OF A4 REQUIREMENTS

A4ROCList = []

def print_ROC_Curve(TITLE, LIST):
    fig = plt.figure(figsize=(6, 4))
    plt.title(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label=theLabel)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def print_Accuracy(TITLE, LIST):
    print(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        ACC = theResults[1]
        print(NAME, " = ", ACC)
    print("------\n")
#
#
# TF: Predicting Loan Defaults with GB-selected variables
def get_TF_ProbAccuracyScores(NAME, MODEL, X, Y):
    probs = MODEL.predict(X)
    pred_list = []
    for p in probs:
        pred_list.append(np.argmax(p))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc]

def TF_printOverfit(NAME, Train, Test):
    print(NAME, "TRAIN AUC: ", Train[4])
    print(NAME, "TEST AUC: ", Test[4])
    print(NAME, "OVERFIT: ", Train[4] - Test[4])

Scaler = MinMaxScaler()
Scaler.fit(x_train)

u_train = Scaler.transform(x_train)
u_test = Scaler.transform(x_test)

u_train = pd.DataFrame(u_train)
u_test = pd.DataFrame(u_test)

u_train.columns = list(x_train.columns.values)
u_test.columns = list(x_test.columns.values)

u_train = u_train[varsGBFlag]
u_test = u_test[varsGBFlag]

# ROUND 1a: RELU, ONE LAYER
print("======================\n\n")
WHO = "Tensor_Flow_F1a"

FShape = u_train.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 500 # iterations
FUnits = int(2*FShape) # nodes

FLay1 = tf.keras.layers.Dense(units = FUnits, activation = FActiv, input_dim = FShape)
FLayOut = tf.keras.layers.Dense(units = 2, activation = tf.keras.activations.softmax)

FCLM = tf.keras.Sequential()
FCLM.add(FLay1)
FCLM.add(FLayOut)
FCLM.compile(loss = FLoss, optimizer = FOptim)
FCLM.fit(u_train, y_train[TargetB], epochs = FEpoch, verbose = False)

F1aCLMTrain = get_TF_ProbAccuracyScores(WHO + "_Train", FCLM, u_train, y_train[TargetB])
F1aCLMTest = get_TF_ProbAccuracyScores(WHO, FCLM, u_test, y_test[TargetB])

print_ROC_Curve(WHO, [F1aCLMTrain, F1aCLMTest])
print_Accuracy(WHO + " F1a CLASSIFICATION ACCURACY", [F1aCLMTrain, F1aCLMTest])
TF_printOverfit("F1a", F1aCLMTrain, F1aCLMTest)

ROCF1a = F1aCLMTest
ROCList.append(ROCF1a)
A4ROCList.append(ROCF1a)

# ROUND 1b: SOFTMAX, ONE LAYER
print("======================\n\n")
WHO = "Tensor_Flow_F1b"

FShape = u_train.shape[1]
FActiv = tf.keras.activations.softmax
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 500 # iterations
FUnits = int(2*FShape) # nodes

FLay1 = tf.keras.layers.Dense(units = FUnits, activation = FActiv, input_dim = FShape)
FLayOut = tf.keras.layers.Dense(units = 2, activation = tf.keras.activations.softmax)

FCLM = tf.keras.Sequential()
FCLM.add(FLay1)
FCLM.add(FLayOut)
FCLM.compile(loss = FLoss, optimizer = FOptim)
FCLM.fit(u_train, y_train[TargetB], epochs = FEpoch, verbose = False)

F1bCLMTrain = get_TF_ProbAccuracyScores(WHO + "_Train", FCLM, u_train, y_train[TargetB])
F1bCLMTest = get_TF_ProbAccuracyScores(WHO, FCLM, u_test, y_test[TargetB])

print_ROC_Curve(WHO, [F1bCLMTrain, F1bCLMTest])
print_Accuracy(WHO + " F1b CLASSIFICATION ACCURACY", [F1bCLMTrain, F1bCLMTest])
TF_printOverfit("F1b", F1bCLMTrain, F1bCLMTest)

ROCF1b = F1bCLMTest
ROCList.append(ROCF1b)
A4ROCList.append(ROCF1b)


# # ROUND 1c: TANH, ONE LAYER
print("======================\n\n")
WHO = "Tensor_Flow_F1c"

FShape = u_train.shape[1]
FActiv = tf.keras.activations.tanh
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 500 # iterations
FUnits = int(2*FShape) # nodes

FLay1 = tf.keras.layers.Dense(units = FUnits, activation = FActiv, input_dim = FShape)
FLayOut = tf.keras.layers.Dense(units = 2, activation = tf.keras.activations.softmax)

FCLM = tf.keras.Sequential()
FCLM.add(FLay1)
FCLM.add(FLayOut)
FCLM.compile(loss = FLoss, optimizer = FOptim)
FCLM.fit(u_train, y_train[TargetB], epochs = FEpoch, verbose = False)

F1cCLMTrain = get_TF_ProbAccuracyScores(WHO + "_Train", FCLM, u_train, y_train[TargetB])
F1cCLMTest = get_TF_ProbAccuracyScores(WHO, FCLM, u_test, y_test[TargetB])

print_ROC_Curve(WHO, [F1cCLMTrain, F1cCLMTest])
print_Accuracy(WHO + " F1c CLASSIFICATION ACCURACY", [F1cCLMTrain, F1cCLMTest])
TF_printOverfit("F1c", F1cCLMTrain, F1cCLMTest)

ROCF1c = F1cCLMTest
ROCList.append(ROCF1c)
A4ROCList.append(ROCF1c)

print_ROC_Curve("F1", [F1aCLMTest, F1bCLMTest, F1cCLMTest])

# ROUND 2: RELU WITH TWO LAYERS
print("======================\n\n")
WHO = "Tensor_Flow_F2"
FShape = u_train.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 500 # iterations
FUnits = int(2*FShape) # nodes

FLay1 = tf.keras.layers.Dense(units = FUnits, activation = FActiv, input_dim = FShape)
FLay2 = tf.keras.layers.Dense(units = FUnits, activation = FActiv)
FLayOut = tf.keras.layers.Dense(units = 2, activation = tf.keras.activations.softmax)

FCLM = tf.keras.Sequential()
FCLM.add(FLay1)
FCLM.add(FLay2)
FCLM.add(FLayOut)
FCLM.compile(loss = FLoss, optimizer = FOptim)
FCLM.fit(u_train, y_train[TargetB], epochs = FEpoch, verbose = False)

F2CLMTrain = get_TF_ProbAccuracyScores(WHO + "_Train", FCLM, u_train, y_train[TargetB])
F2CLMTest = get_TF_ProbAccuracyScores(WHO, FCLM, u_test, y_test[TargetB])

print_ROC_Curve(WHO, [F2CLMTrain, F2CLMTest])
print_Accuracy(WHO + " F2 CLASSIFICATION ACCURACY", [F2CLMTrain, F2CLMTest])
TF_printOverfit("F2", F2CLMTrain, F2CLMTest)

ROCF2 = F2CLMTest
ROCList.append(ROCF2)
A4ROCList.append(ROCF2)


# ROUND 3: RELU WITH 2 LAYERS AND 1 DROP
print("======================\n\n")
WHO = "Tensor_Flow_F3"
FShape = u_train.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 500 # iterations
FUnits = int(2*FShape) # nodes

FLay1 = tf.keras.layers.Dense(units = FUnits, activation = FActiv, input_dim = FShape)
FLayDrop = tf.keras.layers.Dropout(.2)
FLay2 = tf.keras.layers.Dense(units = FUnits, activation = FActiv)
FLayOut = tf.keras.layers.Dense(units = 2, activation = tf.keras.activations.softmax)

FCLM = tf.keras.Sequential()
FCLM.add(FLay1)
FCLM.add(FLayDrop)
FCLM.add(FLay2)
FCLM.add(FLayOut)
FCLM.compile(loss = FLoss, optimizer = FOptim)
FCLM.fit(u_train, y_train[TargetB], epochs = FEpoch, verbose = False)

F3CLMTrain = get_TF_ProbAccuracyScores(WHO + "_Train", FCLM, u_train, y_train[TargetB])
F3CLMTest = get_TF_ProbAccuracyScores(WHO, FCLM, u_test, y_test[TargetB])

print_ROC_Curve(WHO, [F3CLMTrain, F3CLMTest])
print_Accuracy(WHO + " F3 CLASSIFICATION ACCURACY", [F3CLMTrain, F3CLMTest])
TF_printOverfit("F3", F3CLMTrain, F3CLMTest)

ROCF3 = F3CLMTest
ROCList.append(ROCF3)
A4ROCList.append(ROCF3)

# All Models
A4ROCList = sorted(A4ROCList, key = lambda x: x[4], reverse=True)
print_ROC_Curve("ROC TF Flag Models", A4ROCList)

ROCList = sorted(ROCList, key = lambda x: x[4], reverse=True)
print_ROC_Curve("ROC Flag Models", ROCList)



## TF PREDICT LOSS ##
rmses = []

v_train = Scaler.transform(w_train)
v_test = Scaler.transform(w_test)

v_train = pd.DataFrame(v_train)
v_test = pd.DataFrame(v_test)

v_train.columns = list(x.columns.values)
v_test.columns = list(x.columns.values)

# Variable selection with GB-Selected Variables
v_train = v_train[varsGBAmt]
v_test = v_test[varsGBAmt]


# # # ROUND 1a: RELU, ONE LAYER
print("======================\n\n")
WHO = "TF_A1a"

AShape = v_train.shape[1]
AActiv = tf.keras.activations.relu
ALoss = tf.keras.losses.MeanSquaredError()
AOptim = tf.keras.optimizers.Adam()
AEpoch = 1000 # iterations
AUnits = int(2*AShape) # nodes

ALay1 = tf.keras.layers.Dense(units = AUnits, activation = AActiv, input_dim = AShape)
ALayOut = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.linear)

ACLM = tf.keras.Sequential()
ACLM.add(ALay1)
ACLM.add(ALayOut)
ACLM.compile(loss = ALoss, optimizer = AOptim)
ACLM.fit(v_train, z_train[TargetL], epochs = AEpoch, verbose = False)


A1aAMTTrain = getAmtAccuracyScores(WHO + "_Train", ACLM, v_train, z_train[TargetL])
A1aAMTTest = getAmtAccuracyScores(WHO, ACLM, v_test, z_test[TargetL])

print_Accuracy(WHO + " RMSE ACCURACY", [A1aAMTTrain, A1aAMTTest])

rmses.append(A1aAMTTest)
#
#
# # ROUND 1b: LINEAR, 1 LAYER
print("======================\n\n")
WHO = "TF_A1b"

AShape = v_train.shape[1]
AActiv = tf.keras.activations.linear
ALoss = tf.keras.losses.MeanSquaredError()
AOptim = tf.keras.optimizers.Adam()
AEpoch = 1000 # iterations
AUnits = int(2*AShape) # nodes

ALay1 = tf.keras.layers.Dense(units = AUnits, activation = AActiv, input_dim = AShape)
ALayOut = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.linear)

ACLM = tf.keras.Sequential()
ACLM.add(ALay1)
ACLM.add(ALayOut)
ACLM.compile(loss = ALoss, optimizer = AOptim)
ACLM.fit(v_train, z_train[TargetL], epochs = AEpoch, verbose = False)

A1bAMTTrain = getAmtAccuracyScores(WHO + "_Train", ACLM, v_train, z_train[TargetL])
A1bAMTTest = getAmtAccuracyScores(WHO, ACLM, v_test, z_test[TargetL])

print_Accuracy(WHO + " RMSE ACCURACY", [A1bAMTTrain, A1bAMTTest])

rmses.append(A1bAMTTest)



# # ROUND 1c: TANH, 1 LAYER
print("======================\n\n")
WHO = "TF_A1c"

AShape = v_train.shape[1]
AActiv = tf.keras.activations.tanh
ALoss = tf.keras.losses.MeanSquaredError()
AOptim = tf.keras.optimizers.Adam()
AEpoch = 1000 # iterations
AUnits = int(2*AShape) # nodes

ALay1 = tf.keras.layers.Dense(units = AUnits, activation = AActiv, input_dim = AShape)
ALayOut = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.linear)

ACLM = tf.keras.Sequential()
ACLM.add(ALay1)
ACLM.add(ALayOut)
ACLM.compile(loss = ALoss, optimizer = AOptim)
ACLM.fit(v_train, z_train[TargetL], epochs = AEpoch, verbose = False)

A1cAMTTrain = getAmtAccuracyScores(WHO + "_Train", ACLM, v_train, z_train[TargetL])
A1cAMTTest = getAmtAccuracyScores(WHO, ACLM, v_test, z_test[TargetL])

print_Accuracy(WHO + " RMSE ACCURACY", [A1cAMTTrain, A1cAMTTest])

rmses.append(A1cAMTTest)

print("RMSEs A1:")
for i in rmses:
    print(i[0])
    print(i[1])


# # ROUND 2: LINEAR, TWO LAYERS
print("======================\n\n")
WHO = "TF_A2"

AShape = v_train.shape[1]
AActiv = tf.keras.activations.linear
ALoss = tf.keras.losses.MeanSquaredError()
AOptim = tf.keras.optimizers.Adam()
AEpoch = 1000 # iterations
AUnits = int(2*AShape) # nodes

ALay1 = tf.keras.layers.Dense(units = AUnits, activation = AActiv, input_dim = AShape)
ALay2 = tf.keras.layers.Dense(units = AUnits, activation = AActiv)
ALayOut = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.linear)

ACLM = tf.keras.Sequential()
ACLM.add(ALay1)
ACLM.add(ALay2)
ACLM.add(ALayOut)
ACLM.compile(loss = ALoss, optimizer = AOptim)
ACLM.fit(v_train, z_train[TargetL], epochs = AEpoch, verbose = False)

A2AMTTrain = getAmtAccuracyScores(WHO + "_Train", ACLM, v_train, z_train[TargetL])
A2AMTTest = getAmtAccuracyScores(WHO, ACLM, v_test, z_test[TargetL])

print_Accuracy(WHO + " RMSE ACCURACY", [A2AMTTrain, A2AMTTest])

rmses.append(A2AMTTest)


# # ROUND 3: LINEAR, TWO LAYERS, DROPOUT
print("======================\n\n")
WHO = "TF_A3"

AShape = v_train.shape[1]
AActiv = tf.keras.activations.linear
ALoss = tf.keras.losses.MeanSquaredError()
AOptim = tf.keras.optimizers.Adam()
AEpoch = 1000 # iterations
AUnits = int(2*AShape) # nodes

ALay1 = tf.keras.layers.Dense(units = AUnits, activation = AActiv, input_dim = AShape)
ALayDrop = tf.keras.layers.Dropout(.2)
ALay2 = tf.keras.layers.Dense(units = AUnits, activation = AActiv)
ALayOut = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.linear)

ACLM = tf.keras.Sequential()
ACLM.add(ALay1)
ACLM.add(ALayDrop)
ACLM.add(ALay2)
ACLM.add(ALayOut)
ACLM.compile(loss = ALoss, optimizer = AOptim)
ACLM.fit(v_train, z_train[TargetL], epochs = AEpoch, verbose = False)

A3AMTTrain = getAmtAccuracyScores(WHO + "_Train", ACLM, v_train, z_train[TargetL])
A3AMTTest = getAmtAccuracyScores(WHO, ACLM, v_test, z_test[TargetL])

print_Accuracy(WHO + " RMSE ACCURACY", [A3AMTTrain, A3AMTTest])

rmses.append(A3AMTTest)






for i in rmses:
    rmsesall.append(i)

dflist = []
for i in rmsesall:
    dflist.append([i[0], i[1]])

rmsesDF = pd.DataFrame(dflist, columns = ["Model", "RMSE"])

print("All Model RMSEs Comparison:")
print(rmsesDF)
