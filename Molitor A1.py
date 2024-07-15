import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Add stuff to histograms to format/visualize better
sns.set()  # So it magically does it

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

infile = "C:\\Users\\ajmol\\Documents\\Programming\\ML\\HMEQ_Loss.csv"

df = pd.read_csv(infile)

print(df.head().T)
print()
print(df.describe().T)
print()

dt = df.dtypes

objList = []  # strings
intList = []
floatList =[]

TargetL = "TARGET_LOSS_AMT"
TargetB = "TARGET_BAD_FLAG"

print("EXPLORATORY ANALYSIS:\n")

for i in dt.index:
    if i in ([TargetB, TargetL]): continue
    if dt[i] in (["object"]) : objList.append(i)
    if dt[i] in (["int64"]) : intList.append(i)
    if dt[i] in (["float64"]) : floatList.append(i)

print("Exploratory Analysis of Object Variables:\n")
for i in objList:
    print(" Class = ", i)
    g = df.groupby(i)
    print(g[i].count(), "\n")
    x = g[TargetB].mean()
    print("Bad Loan Prob", x)
    x = round(g[TargetL].mean(),2)
    print("Avg Loan Loss Amount", x, "\n\n")

print("Exploratory Analysis of Float Variables:\n")
for i in floatList:
    print("Variable:", i)
    g = df.groupby(TargetB)
    x = g[i].mean()
    print("Average {} based on Loan Flag:".format(i))
    print(x)
    c = df[i].corr(df[TargetB])
    c = round(100*c, 1)
    print("Loan Flag Correlation:", c, "%")
    c = df[i].corr(df[TargetL])
    c = round(100*c, 1)
    print("Loan Loss Correlation:", c, "%\n")

for i in intList:
    print("Variable:", i)
    g = df.groupby(TargetB)
    x = g[i].mean()
    print("Bad Loan Prob:", x)
    c = df[i].corr(df[TargetB])
    c = round(100*c, 1)
    print("Loan Flag Correlation:", c, "%")
    c = df[i].corr(df[TargetL])
    c = round(100*c, 1)
    print("Loan Loss Correlation:", c, "\n")

print("Exploratory Analysis of Target Variables:")

for i in ([TargetB, TargetL]):
    print(df[i].describe())
    print("Mode: ", df[i].mode())
    print("Median: ", df[i].median())

print()

print("GRAPHICAL EXPLORATION:\n")
#
# for i in objList:  # Cycle through individual pie charts for each categorical variable
#     x = df[i].value_counts(dropna = False)
#     theLabels = x.axes[0].tolist()
#     theSlices = list(x)
#     plt.pie(theSlices,
#             labels = theLabels,
#             startangle = 90,
#             shadow=False,
#             autopct = "%1.1f%%")
#     plt.title("Pie Chart: " + i)
#     plt.show()
#
# for i in floatList:
#     plt.hist(df[i])
#     plt.xlabel(i)
#     plt.ylabel("Frequency")
#     plt.title("Histogram of " + i)
#     plt.show()
#
# for i in intList:
#     plt.hist(df[i])
#     plt.xlabel(i)
#     plt.ylabel("Frequency")
#     plt.title("Histogram of " + i)
#     plt.show()
#
# plt.hist(df[TargetL])
# plt.xlabel("Loan Loss Amounts")
# plt.ylabel("Frequency")
# plt.title("Loan Loss Histogram")
# plt.show()
#
# plt.hist(df["MORTDUE"]/df["VALUE"])
# plt.xlabel("Mortgage Vs. House Value")
# plt.ylabel("Frequency")
# plt.title("Mortgage to Value Ratio Histogram")
# plt.show()

# boxplot = sns.boxplot(data=df, x = df["JOB"], y = df[TargetL])
# boxplot.set_xticklabels(boxplot.get_xticklabels(), rotation =40)
# plt.title("Loan Loss Amounts Based on Job")
# plt.show()
#
# boxplot = sns.boxplot(data=df, x = df[TargetL])
# plt.title("Loan Loss For Defaulted Loans")
# plt.show()

print("IMPUTE MISSING DATA:\n")

for i in objList:
    print(i)
    g = df.groupby(i)
    print(g[i].count())
    print("Most Common: ", df[i].mode()[0])
    print("Missing: ", df[i].isna().sum(), "\n")

for i in objList:
    if df[i].isna().sum() == 0: continue
    NAME = "IMP_"+i
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    g = df.groupby(NAME)
    df = df.drop(i, axis = 1)

print("ONE HOT ENCODING:\n")

print(df.head().T)

df["z_IMP_REASON_HomeImp"] = (df["IMP_REASON"].isin(["HomeImp"]) + 0)
df["z_IMP_REASON_DebtCon"] = (df["IMP_REASON"].isin(["DebtCon"]) + 0)
# Both false = Reason MISSING

df["z_IMP_JOB_Mgr"] = (df["IMP_JOB"].isin(["Mgr"]) + 0)
df["z_IMP_JOB_Office"] = (df["IMP_JOB"].isin(["Office"]) + 0)
df["z_IMP_JOB_Other"] = (df["IMP_JOB"].isin(["Other"]) + 0)
df["z_IMP_JOB_ProfExe"] = (df["IMP_JOB"].isin(["ProfExe"]) + 0)
df["z_IMP_JOB_Sales"] = (df["IMP_JOB"].isin(["Sales"]) + 0)
df["z_IMP_JOB_Self"] = (df["IMP_JOB"].isin(["Self"]) + 0)
# All false = Job MISSING

df = df.drop("IMP_REASON", axis = 1)
df = df.drop("IMP_JOB", axis = 1)

print("\n\n\n")
dt = df.dtypes
numList = []
for i in dt.index:
    if i in ([TargetB, TargetL]): continue
    if dt[i] in (["float64", "int64"]): numList.append(i)

for i in numList:
    if df[i].isna().sum() == 0: continue
    flag = "M_" + i
    imp = "IMP_" + i
    df[flag] = df[i].isna() + 0
    df[imp] = df[i]
    df.loc[df[imp].isna(), imp] = df[i].median()
    df = df.drop(i, axis = 1)

print(df.head(10).T)
