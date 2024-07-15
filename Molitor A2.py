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
print("FLAG TRAINING:", x_train.shape)
print("FLAG TEST:", x_test.shape)

dectree = tree.DecisionTreeClassifier(max_depth = 4)
dectree = dectree.fit(x_train, y_train[TargetB])

yPredTrain = dectree.predict(x_train)
yPredTest = dectree.predict(x_test)

print()
print("Train Accuracy:", metrics.accuracy_score(y_train[TargetB], yPredTrain))
print("Test Accuracy:", metrics.accuracy_score(y_test[TargetB], yPredTest))

# ROC CURVE
probs = dectree.predict_proba(x_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
ROCAucTrain = metrics.auc(fpr_train, tpr_train)  # Area under curve = AUC
# Higher = more accurate

probs = dectree.predict_proba(x_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
ROCAucTest = metrics.auc(fpr_test, tpr_test)

print()
print("ROC AUC Train:", ROCAucTrain)
print("ROC AUC Test:", ROCAucTest)

# ROC GRAPH
plt.title('TREE ROC CURVE')
plt.plot(fpr_train, tpr_train, "b", label = "AUC TRAIN = %0.2f" % ROCAucTrain, color = "blue")
plt.plot(fpr_test, tpr_test, "b", label = "AUC TEST = %0.2f" % ROCAucTest, color = "red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# plt.show()


# GRAPHVIZ
feature_cols = list(x.columns.values)
# tree.export_graphviz(dectree, out_file = "HMEQ_DecTreeN.txt", filled=True, rounded = True, feature_names = feature_cols,
#                      impurity = False, class_names = ["Good", "Bad"])

fpr_tree = fpr_test
tpr_tree = tpr_test
auc_tree = ROCAucTest

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

print()
print("See bottom of file for extra decision tree fun.")

# DECISION TREE FOR LOSS AMOUNT


F = ~ y_train[TargetL].isna()
w_train = x_train[F]
z_train = y_train[F]

F = ~ y_test[TargetL].isna()
w_test = x_test[F]
z_test = y_test[F]

print(w_test.head().T)
print(z_test.head().T)

# DAMAGES REGRESSION TREE
amtTree = tree.DecisionTreeRegressor(max_depth =3)
amtTree = amtTree.fit(w_train, z_train[TargetL])

zPredTrain = amtTree.predict(w_train)
zPredTest = amtTree.predict(w_test)


RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(z_train[TargetL], zPredTrain))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(z_test[TargetL], zPredTest))  # error in # of dollars

print("TREE RMSE TRAIN: $", RMSE_TRAIN)
print("TREE RMSE TEST: $", RMSE_TEST)
print()

RMSE_TREE = RMSE_TEST

feature_cols = list(x.columns.values)
vars_tree_amt = getTreeVars(amtTree, feature_cols)
# tree.export_graphviz(amtTree, out_file = 'HMEQ_DecTree_amtN.txt',filled=True, rounded = True, feature_names = feature_cols,
#                      impurity = False, precision=0)

print("Decision Tree Regression predictive variables:")
for i in vars_tree_amt:
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

# ROC for Random Forest
probs = dectreeRF.predict_proba(x_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve( y_train[TargetB], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = dectreeRF.predict_proba(x_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_RF = fpr_test
tpr_RF = tpr_test
auc_RF = roc_auc_test

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
vars_RF_flag = getEnsembleTreeVars(dectreeRF, feature_cols)

print("Random Forest Classifier tree predictive variables:")
for i in vars_RF_flag :
   print( i )

plt.title('RF ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()

# RF Regressor to Predict Damages
RegRF = RandomForestRegressor(n_estimators = 100, random_state=1)
RegRF = RegRF.fit(w_train, z_train[TargetL])

Z_Pred_train = RegRF.predict(w_train)
Z_Pred_test = RegRF.predict(w_test)

RMSE_TRAIN = math.sqrt( metrics.mean_squared_error(z_train[TargetL], Z_Pred_train))
RMSE_TEST = math.sqrt( metrics.mean_squared_error(z_test[TargetL], Z_Pred_test))

print("RF RMSE Train: $", RMSE_TRAIN)
print("RF RMSE Test: $", RMSE_TEST)

RMSE_RF = RMSE_TEST

feature_cols = list(x.columns.values )
vars_RF_amt = getEnsembleTreeVars(RegRF, feature_cols )

print("Random Forest regression predictive variables:")
for i in vars_RF_amt :
   print( i )


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

probs = dectreeGB.predict_proba(x_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(y_train[TargetB], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

probs = dectreeGB.predict_proba(x_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(y_test[TargetB], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

fpr_GB = fpr_test
tpr_GB = tpr_test
auc_GB = roc_auc_test


feature_cols = list(x.columns.values)
vars_GB_flag = getEnsembleTreeVars(dectreeGB, feature_cols)

print("Gradient Boost Classifier Tree predictive variables:")
for i in vars_GB_flag :
   print(i)


plt.title('GB ROC CURVE')
plt.plot(fpr_train, tpr_train, label = 'AUC TRAIN = %0.2f' % roc_auc_train, color="blue")
plt.plot(fpr_test, tpr_test, label = 'AUC TEST = %0.2f' % roc_auc_test, color="red")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()



# GB Regression

RegGB = GradientBoostingRegressor(random_state=1)
RegGB = RegGB.fit(w_train, z_train[TargetL])

Z_Pred_train = RegGB.predict(w_train)
Z_Pred_test = RegGB.predict(w_test)

RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(z_train[TargetL], Z_Pred_train))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(z_test[TargetL], Z_Pred_test))

print("GB RMSE Train: $", RMSE_TRAIN )
print("GB RMSE Test: $", RMSE_TEST )

RMSE_GB = RMSE_TEST

feature_cols = list(x.columns.values )
vars_GB_amt = getEnsembleTreeVars(RegGB, feature_cols)

print("Gradient Boost regression model predictive variables:")
for i in vars_GB_amt:
   print(i)


# ROC for all models
plt.title('MODELS ROC CURVE')
plt.plot(fpr_tree, tpr_tree, label = 'AUC TREE = %0.2f' % auc_tree, color="red")
plt.plot(fpr_RF, tpr_RF, label = 'AUC RF = %0.2f' % auc_RF, color="green")
plt.plot(fpr_GB, tpr_GB, label = 'AUC GB = %0.2f' % auc_GB, color="blue")
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# plt.show()


print()
print("Root Mean Square Average For Damages")
print("TREE", round(RMSE_TREE, 2))
print("RF", round(RMSE_RF, 2))
print("GB", round(RMSE_GB, 2))







# Extra bingo bonus fun things: Looping through the first 5 random seeds of the decision tree to get the average
# model accuracy and find the common predictor variables in each model:


testacc = []
testvars = []

for i in [1, 2, 3, 4, 5]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8, test_size = .2, random_state=i)

    dectree = tree.DecisionTreeClassifier(max_depth = 4)
    dectree = dectree.fit(x_train, y_train[TargetB])

    yPredTrain = dectree.predict(x_train)
    yPredTest = dectree.predict(x_test)

    testacc.append(round(metrics.accuracy_score(y_test[TargetB], yPredTest), 3))

    feature_cols = list(x.columns.values)
    varsTreeFlag = getTreeVars(dectree, feature_cols)

    testvars.append(varsTreeFlag)

print()
nolist = []
fulllist = []
count = 0
for j in [0, 1, 2, 3, 4]:
    for i in testvars[j]:
        if i not in fulllist and i not in nolist:
            fulllist.append(i)
        for l in fulllist:
            if l in fulllist and l not in testvars[j]:
                fulllist.remove(l)
                nolist.append(l)

print()
# print(fulllist)
for i in fulllist:
    if i not in testvars[0]:
        nolist.append(i)

for i in nolist:
    if i in fulllist:
        fulllist.remove(i)

print("Key predictive variables in each of the first five models:", fulllist)
print("Number of key variables present in some but not all models:", len(nolist))

avgacc = sum(testacc)/len(testacc)
print("Average first 5 model accuracy of test data:", avgacc)
print("Accuracy of first test model with random seed 1:", round(auc_tree, 2))

# The variables are basically almost the same as those in the first random seed. But, they do differ among seed, so this
# similarity to the first seed is likely chance.
