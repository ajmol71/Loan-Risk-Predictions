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
from sklearn.ensemble import  GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

infile = "C:\\Users\\ajmol\\Documents\\Programming\\ML\\HMEQ_Loss_Imputed.csv"

df = pd.read_csv(infile)
df = df.drop("Unnamed: 0", axis = 1)

print(df.head().T)

TargetL = "TARGET_LOSS_AMT"
TargetB = "TARGET_BAD_FLAG"

dt = df.dtypes

# SPLIT DATA

x = df.copy()
x = x.drop(TargetL, axis = 1)
x = x.drop(TargetB, axis = 1)
y = df[[TargetB, TargetL]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8, test_size = .2, random_state=1)

print()
print("x train:")
print(x_train.head().T)
print()
print("y train:")
print(y_train.head().T)

# DECISION TREE
dectree = tree.DecisionTreeClassifier(max_depth = 4)
dectree = dectree.fit(x_train, y_train[TargetB])

yPredTrain = dectree.predict(x_train)
yPredTest = dectree.predict(x_test)

print()
print("Train Accuracy:", metrics.accuracy_score(y_train[TargetB], yPredTrain))
print("Test Accuracy:", metrics.accuracy_score(y_test[TargetB], yPredTest))

feature_cols = list(x.columns.values)

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

varsTreeFlag = getTreeVars(dectree, feature_cols)

print()
print("Decision Tree Classifier predictive variables:")
for i in varsTreeFlag:
    print(i)

# DECISION TREE FOR LOSS AMOUNT


F = ~ y_train[TargetL].isna()
w_train = x_train[F]
z_train = y_train[F]

F = ~ y_test[TargetL].isna()
w_test = x_test[F]
z_test = y_test[F]

# DAMAGES REGRESSION TREE
amtTree = tree.DecisionTreeRegressor(max_depth =3)
amtTree = amtTree.fit(w_train, z_train[TargetL])

zPredTrain = amtTree.predict(w_train)
zPredTest = amtTree.predict(w_test)

RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(z_train[TargetL], zPredTrain))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(z_test[TargetL], zPredTest))  # error in # of dollars

print()
print("TREE RMSE TRAIN: $", RMSE_TRAIN)
print("TREE RMSE TEST: $", RMSE_TEST)
print()

feature_cols = list(x.columns.values)
varsTreeAmt = getTreeVars(amtTree, feature_cols)

print("Decision Tree Regression predictive variables:")
for i in varsTreeAmt:
    print(i)

##  RANDOM FOREST  ##

dectreeRF = RandomForestClassifier(n_estimators = 100, random_state=1)
dectreeRF = dectreeRF.fit(x_train, y_train[TargetB])

y_pred_train = dectreeRF.predict(x_train)
y_pred_test = dectreeRF.predict(x_test)

print()
print("RF Prob of Default:")
print("Train Accuracy:", metrics.accuracy_score(y_train[TargetB], y_pred_train))
print("Test Accuracy:", metrics.accuracy_score(y_test[TargetB], y_pred_test))
print()

# What vars were most common for the RF?
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


feature_cols = list(x.columns.values)
varsRFFlag1 = getEnsembleTreeVars(dectreeRF, feature_cols)
varsRFFlag = []
for i in varsRFFlag1:
    varsRFFlag.append(i[0])

print("Random Forest Classifier tree predictive variables:")
for i in varsRFFlag:
   print(i)

# RF Regressor to Predict Damages
RegRF = RandomForestRegressor(n_estimators = 100, random_state=1)
RegRF = RegRF.fit(w_train, z_train[TargetL])

Z_Pred_train = RegRF.predict(w_train)
Z_Pred_test = RegRF.predict(w_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(z_train[TargetL], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(z_test[TargetL], Z_Pred_test))

print()
print("RF RMSE Train: $", RMSE_TRAIN)
print("RF RMSE Test: $", RMSE_TEST)

feature_cols = list(x.columns.values )
varsRFAmt1 = getEnsembleTreeVars(RegRF, feature_cols)

varsRFAmt = []
for i in varsRFAmt1:
    varsRFAmt.append(i[0])

print("Random Forest regression predictive variables:")
for i in varsRFAmt:
   print(i)


# GRADIENT BOOSTING #

dectreeGB = GradientBoostingClassifier( random_state=1 )
dectreeGB = dectreeGB.fit(x_train, y_train[TargetB])

Y_Pred_train = dectreeGB.predict(x_train)
Y_Pred_test = dectreeGB.predict(x_test)

print("\n=============\n")
print("GRADIENT BOOSTING\n")
print("Probability of crash")
print("Accuracy Train:",metrics.accuracy_score(y_train[TargetB], Y_Pred_train))
print("Accuracy Test:",metrics.accuracy_score(y_test[TargetB], Y_Pred_test))
print("\n")

feature_cols = list(x.columns.values)
varsGBFlag1 = getEnsembleTreeVars(dectreeGB, feature_cols)

varsGBFlag = []
for i in varsGBFlag1:
    varsGBFlag.append(i[0])

print("Gradient Boost Classifier Tree predictive variables:")
for i in varsGBFlag:
   print(i)


# GB Regression

RegGB = GradientBoostingRegressor(random_state=1)
RegGB = RegGB.fit(w_train, z_train[TargetL])

Z_Pred_train = RegGB.predict(w_train)
Z_Pred_test = RegGB.predict(w_test)

RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(z_train[TargetL], Z_Pred_train))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(z_test[TargetL], Z_Pred_test))

print()
print("GB RMSE Train: $", RMSE_TRAIN )
print("GB RMSE Test: $", RMSE_TEST )
print()

feature_cols = list(x.columns.values )
varsGBAmt1 = getEnsembleTreeVars(RegGB, feature_cols)

varsGBAmt = []
for i in varsGBAmt1:
    varsGBAmt.append(i[0])

print("Gradient Boost regression model predictive variables:")
for i in varsGBAmt:
   print(i)





### NEW A3 CODE ###

# Logistic Regression All Variables
print("\n\n\n")
m = LogisticRegression()
m.fit(x_train, y_train[TargetB])

varNames = list(x_train.columns.values)
coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varNames):
    coefdict[feat] = coef

print()
print("Logistic Model All Vars")
for i in coefdict:
    print(i, "=", coefdict[i])

predflagtrain = m.predict(x_train)
predflagtest = m.predict(x_test)

predprobtrain = m.predict_proba(x_train)  # predicted 0 and 1 probabilities
p1 = predprobtrain[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
aucAllTrain = metrics.auc(fpr_train, tpr_train)

predprobtest = m.predict_proba(x_test)
p1 = predprobtest[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
aucAllTest = metrics.auc(fpr_test, tpr_test)

print()
print("AUC All Var Logistic Train:", aucAllTrain)
print("AUC All Var Logistic Test:", aucAllTest)

# ROC GRAPH
plt.title('REGRESSION ALL ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % aucAllTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % aucAllTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

fpr_all = fpr_test
tpr_all = tpr_test
auc_all = aucAllTest

accAllTrain = metrics.accuracy_score(y_train[TargetB], predflagtrain)
accAllTest = metrics.accuracy_score(y_test[TargetB], predflagtest)

print()
print("Accuracy All Var Logistic Train:", accAllTrain)
print("Accuracy All Var Logistic Test:", accAllTest)



# Logistic - Decision Tree Classifier
m_train = x_train[varsTreeFlag]
m_test = x_test[varsTreeFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsTreeFlag):
    coefdict[feat] = coef

print()
print("Logistic Model from Decision Tree Classifier Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predflagtrain = m.predict(m_train)
predflagtest = m.predict(m_test)

predprobtrain = m.predict_proba(m_train)  # predicted 0 and 1 probabilities
p1 = predprobtrain[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
aucTreeTrain = metrics.auc(fpr_train, tpr_train)

predprobtest = m.predict_proba(m_test)
p1 = predprobtest[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
aucTreeTest = metrics.auc(fpr_test, tpr_test)

print()
print("AUC Tree Var Logistic Train:", aucTreeTrain)
print("AUC Tree Var Logistic Test:", aucTreeTest)

# ROC GRAPH
plt.title('LOGISTIC TREE ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % aucTreeTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % aucTreeTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

fpr_tree = fpr_test
tpr_tree = tpr_test
auc_tree = aucTreeTest

accTreeTrain = metrics.accuracy_score(y_train[TargetB], predflagtrain)
accTreeTest = metrics.accuracy_score(y_test[TargetB], predflagtest)

print()
print("Accuracy Tree Var Logistic: Train", accTreeTrain)
print("Accuracy Tree Var Logistic Test:", accTreeTest)



# Logistic - RF Classifier
print()
m_train = x_train[varsRFFlag]
m_test = x_test[varsRFFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsRFFlag):
    coefdict[feat] = coef

print()
print("Logistic Model from Random Forest Classifier Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predflagtrain = m.predict(m_train)
predflagtest = m.predict(m_test)

predprobtrain = m.predict_proba(m_train)
p1 = predprobtrain[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
aucRFTrain = metrics.auc(fpr_train, tpr_train)

predprobtest = m.predict_proba(m_test)
p1 = predprobtest[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
aucRFTest = metrics.auc(fpr_test, tpr_test)

print()
print("AUC RF Var Logistic Train:", aucRFTrain)
print("AUC RF Var Logistic Test:", aucRFTest)

# ROC GRAPH
plt.title('LOGISTIC RF ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % aucRFTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % aucRFTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

fpr_RF = fpr_test
tpr_RF = tpr_test
auc_RF = aucRFTest

accRFTrain = metrics.accuracy_score(y_train[TargetB], predflagtrain)
accRFTest = metrics.accuracy_score(y_test[TargetB], predflagtest)

print()
print("Accuracy RF Logistic: Train", accRFTrain)
print("Accuracy RF Logistic Test:", accRFTest)

# Logistic - GB Classifier
print()
m_train = x_train[varsGBFlag]
m_test = x_test[varsGBFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsGBFlag):
    coefdict[feat] = coef

print()
print("Logistic Model from Gradient Boosting Classifier Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predflagtrain = m.predict(m_train)
predflagtest = m.predict(m_test)

predprobtrain = m.predict_proba(m_train)
p1 = predprobtrain[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
aucGBTrain = metrics.auc(fpr_train, tpr_train)

predprobtest = m.predict_proba(m_test)
p1 = predprobtest[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
aucGBTest = metrics.auc(fpr_test, tpr_test)

print()
print("AUC GB Var Logistic Train:", aucGBTrain)
print("AUC GB Var Logistic Test:", aucGBTest)

# ROC GRAPH
plt.title('LOGISTIC GB ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % aucGBTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % aucGBTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

fpr_GB = fpr_test
tpr_GB = tpr_test
auc_GB = aucGBTest

accGBTrain = metrics.accuracy_score(y_train[TargetB], predflagtrain)
accGBTest = metrics.accuracy_score(y_test[TargetB], predflagtest)

print()
print("Accuracy GB Logistic: Train", accGBTrain)
print("Accuracy GB Logistic Test:", accGBTest)


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


print("Logistic Forward Selection Predictive Variables")
for i in varsFSFlag:
    print(i)

print()
m_train = x_train[varsFSFlag]
m_test = x_test[varsFSFlag]
m = LogisticRegression()
m.fit(m_train, y_train[TargetB])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0], varsFSFlag):
    coefdict[feat] = coef

print()
print("Logistic Model from Forward Selection Classifier Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predflagtrain = m.predict(m_train)
predflagtest = m.predict(m_test)

predprobtrain = m.predict_proba(m_train)
p1 = predprobtrain[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
aucFSTrain = metrics.auc(fpr_train, tpr_train)

predprobtest = m.predict_proba(m_test)
p1 = predprobtest[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
aucFSTest = metrics.auc(fpr_test, tpr_test)

print()
print("AUC FS Var Logistic Train:", aucFSTrain)
print("AUC FS Var Logistic Test:", aucFSTest)

# ROC GRAPH
plt.title('LOGISTIC FS ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % aucFSTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % aucFSTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

fpr_FS = fpr_test
tpr_FS = tpr_test
auc_FS = aucFSTest

accFSTrain = metrics.accuracy_score(y_train[TargetB], predflagtrain)
accFSTest = metrics.accuracy_score(y_test[TargetB], predflagtest)

print()
print("Accuracy FS Logistic: Train", accFSTrain)
print("Accuracy FS Logistic Test:", accFSTest)




# All Logistic Models
plt.title('LOGISTIC MODELS ROC CURVE')
plt.plot(fpr_all, tpr_all, label = 'AUC ALL = %0.2f' % auc_all, color = "orange")
plt.plot(fpr_tree, tpr_tree, label = 'AUC TREE = %0.2f' % auc_tree, color="red")
plt.plot(fpr_RF, tpr_RF, label = 'AUC RF = %0.2f' % auc_RF, color="green")
plt.plot(fpr_GB, tpr_GB, label = 'AUC GB = %0.2f' % auc_GB, color="blue")
plt.plot(fpr_FS, tpr_FS, label = 'AUC FS = %0.2f' % auc_FS, color="purple")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



### LINEAR REGRESSION MODELS ###

print("\n\n\n")
m = LinearRegression()
m.fit(w_train, z_train[TargetL])

varNames = list(w_train.columns.values)
coefdict = {}
coefdict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_, varNames):
    coefdict[feat] = coef

print()
print("Linear Model All Vars")
for i in coefdict:
    print(i, "=", coefdict[i])

predLossTrain = m.predict(w_train)
predLossTest = m.predict(w_test)

rmseAllTrain = metrics.mean_squared_error(z_train[TargetL], predLossTrain, squared = False)
rmseAllTest = metrics.mean_squared_error(z_test[TargetL], predLossTest, squared = False)

print()
print("RMSE All Vars Linear Train:", rmseAllTrain)
print("RMSE All Vars Linear Test:", rmseAllTest)


# Linear - Dec Tree Vars
m_train = w_train[varsTreeAmt]
m_test = w_test[varsTreeAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_, varsTreeAmt):
    coefdict[feat] = coef

print()
print("Linear Model from Decision Tree Regression Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predTreeTrain = m.predict(m_train)
predTreeTest = m.predict(m_test)

rmseTreeTrain = metrics.mean_squared_error(z_train[TargetL], predTreeTrain, squared = False)
rmseTreeTest = metrics.mean_squared_error(z_test[TargetL], predTreeTest, squared = False)

print()
print("RMSE Tree Vars Linear Train:", rmseTreeTrain)
print("RMSE Tree Vars Linear Test:", rmseTreeTest)


# Linear - RF Vars
m_train = w_train[varsRFAmt]
m_test = w_test[varsRFAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_, varsRFAmt):
    coefdict[feat] = coef

print()
print("Linear Model from Random Forest Regression Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predRFTrain = m.predict(m_train)
predRFTest = m.predict(m_test)

rmseRFTrain = metrics.mean_squared_error(z_train[TargetL], predRFTrain, squared = False)
rmseRFTest = metrics.mean_squared_error(z_test[TargetL], predRFTest, squared = False)

print()
print("RMSE RF Vars Linear Train:", rmseRFTrain)
print("RMSE RF Vars Linear Test:", rmseRFTest)



# Linear - GB Vars
m_train = w_train[varsGBAmt]
m_test = w_test[varsGBAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_, varsGBAmt):
    coefdict[feat] = coef

print()
print("Linear Model from Gradient Boosting Regression Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predGBTrain = m.predict(m_train)
predGBTest = m.predict(m_test)

rmseGBTrain = metrics.mean_squared_error(z_train[TargetL], predGBTrain, squared = False)
rmseGBTest = metrics.mean_squared_error(z_test[TargetL], predGBTest, squared = False)

print()
print("RMSE GB Vars Linear Train:", rmseGBTrain)
print("RMSE GB Vars Linear Test:", rmseGBTest)



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


print("Linear Forward Selection Predictive Variables")
for i in varsFSAmt:
    print(i)

print()
m_train = w_train[varsFSAmt]
m_test = w_test[varsFSAmt]
m = LinearRegression()
m.fit(m_train, z_train[TargetL])

coefdict = {}
coefdict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_, varsFSAmt):
    coefdict[feat] = coef

print()
print("Linear Model from Forward Selection Regression Variables:")
for i in coefdict:
    print(i, "=", coefdict[i])

predFSTrain = m.predict(m_train)
predFSTest = m.predict(m_test)

rmseFSTrain = metrics.mean_squared_error(z_train[TargetL], predFSTrain, squared = False)
rmseFSTest = metrics.mean_squared_error(z_test[TargetL], predFSTest, squared = False)

print()
print("RMSE FS Vars Linear Train:", rmseFSTrain)
print("RMSE FS Vars Linear Test:", rmseFSTest)

rmses = pd.DataFrame(np.array([[round(rmseAllTrain, 2), round(rmseTreeTrain, 2), round(rmseRFTrain, 2),
                            round(rmseGBTrain, 2), round(rmseFSTrain)], [round(rmseAllTest), round(rmseTreeTest),
                            round(rmseRFTest), round(rmseGBTest), round(rmseFSTest)]]), columns = ["All", "Tree", "RF",
                            "GB", "FS"], index = ["Train", "Test"])

print("\n\n")
print("RMSEs for Linear Models:")
print(rmses)

