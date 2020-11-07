import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np

file = "finalData.csv"


# method for initializing .csv file
def initCSV(csvfile):
    df = pd.read_csv(csvfile)

    # change width so all columns can be seen
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 15)

    return df


# function to drop a column in the list cols
def drop(df, cols):

    print("\nDroppable Columns:")
    for i in range(len(cols)):
        print(str(i+1) + ". " + str(cols[i]))

    while True:
        try:
            ans = int(input("Choose a column to drop by entering the corresponding number (or 0 to cancel): "))
            if ans not in range(0, len(cols)+1):
                raise ValueError
        except ValueError:
            print("Error: please enter the number of an option above.\n")
        else:
            break

    if ans == 1:
        newdf = df.drop(cols[0], axis=1)
        print("\nDropped column: " + cols[0])
        del(cols[0])

    elif ans == 2:
        newdf = df.drop(cols[1], axis=1)
        print("\nDropped column: " + cols[1])
        del (cols[1])

    elif ans == 3:
        newdf = df.drop(cols[2], axis=1)
        print("\nDropped column: " + cols[2])
        del (cols[2])

    elif ans == 4:
        newdf = df.drop(cols[3], axis=1)
        print("\nDropped column: " + cols[3])
        del (cols[3])

    elif ans == 5:
        newdf = df.drop(cols[4], axis=1)
        print("\nDropped column: " + cols[4])
        del (cols[4])
    else:
        print("Colummn drop cancelled.")
        return df

    print("New Dataset: ")
    print(newdf)

    while len(cols) > 1:
        try:
            ans = input("Would you like to drop another column?\nEnter 'y' or 'n': ")
            if ans not in ('y', 'Y', 'n', 'N'):
                raise ValueError
        except ValueError:
            print("Error: please enter 'y' for yes or 'n' for no.\n")
        else:
            break

    if ans in ('y', 'Y'):
        drop(newdf, cols)
    elif ans in ('n', 'N'):
        pass
    return newdf


# function for determining the correlation percentages between the various columns of a dataframe (df)
# low = maximum tolerance of correlation, any correlation percent above the low value is flagged
def correlation(df):
    # establish new dataframe based on the correlation of df
    corrDF = df.corr()
    # convert correlation dataframe to numpy array
    corrArr = corrDF.to_numpy()
    # list to hold the values of correlation
    corrValueList = []
    # list to hold the column relationships flagged for high correlation
    flaggedList = []
    # list to hold unique flagged columns
    uniqueFlagged = []

    print("\nCorrelation values:")
    print(corrDF)

    while True:
        try:
            low = float(input("\nEnter the maximum correlation tolerance as a decimal percentage between 0 and 1: "))
            if low < 0 or low >= 1:
                raise ValueError
        except ValueError:
            print("Error: please enter a decimal percentage between 0 and 1 (exclusive).")
        else:
            break

    # iterate through each cell in dataframe
    for i in range(corrArr.shape[0]):
        for j in range(corrArr.shape[1]):
            # if the correlation value is greater than the low threshold and less than 1
            if low < corrArr[i, j] < 1:
                # flag the two columns of correlation
                flagged = (corrDF.columns[i], corrDF.columns[j])
                # check for the reverse pairing as well
                revintersect = (corrDF.columns[j], corrDF.columns[i])
                # if the flagged pair is not yet identified
                if (flagged not in flaggedList) and (revintersect not in flaggedList):
                    # append the flagged pair
                    flaggedList.append(flagged)
                    # append the correlation value of the flagged pair
                    corrValueList.append(corrArr[i, j])

    # convert flagged list to numpy array for easier management
    flaggedArr = np.array(flaggedList)
    # for each row in flagged array
    for i in range(flaggedArr.shape[0]):
        if flaggedArr[i, 0] not in uniqueFlagged:
            uniqueFlagged.append(flaggedArr[i, 0])
        if flaggedArr[i, 1] not in uniqueFlagged:
            uniqueFlagged.append(flaggedArr[i, 1])

    print("Flagged columns with a correlation greater than " + str(low) + ":")
    return drop(df, uniqueFlagged)


data = initCSV(file)
print("Original Dataset:")
print(data)

# newData = correlation(data)

# X for attributes
# Y for target value (label)
attributes = data.drop('quality', axis=1)
label = data['quality']
qualities = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


le = preprocessing.LabelEncoder()

# X for attributes
# Y for target value (label)
attributes = data.drop('Action', axis=1)
label = data['Action']
code = le.fit(['allow', 'deny', 'drop', 'reset-both'])
qualities = le.transform(['allow', 'deny', 'drop', 'reset-both'])

# print(data.describe())


# function to split data into training and testing for tree. test size is 1/5 of the data
def split(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)

    return xtrain, xtest, ytrain, ytest


# function to make a classifier decision tree from data
def treeClassify(xtrain, xtest, ytrain, ytest, plot):
    clf = tree.DecisionTreeClassifier()
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)


    # if plot is true, export as visualized tree
    if plot:
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=attributes.columns, class_names=qualities, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(file + "_Classifier")
        print("Exported classifier tree to pdf")
    # return prediction accuracy of algorithm
    return pd.DataFrame({'Actual': ytest, 'Predicted': ypred})


# function to make a regression decision tree from data
def treeRegression(xtrain, xtest, ytrain, ytest, plot):
    reg = tree.DecisionTreeRegressor()      # initialize regression tree
    reg = reg.fit(xtrain, ytrain)                 # fit
    ypred = reg.predict(xtest)

    # if plot is true, export as visualized tree
    if plot:
        dot_data = tree.export_graphviz(reg, out_file=None, feature_names=attributes.columns, class_names=qualities, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(file + "_Regression")
        print("Exported regression tree to pdf")
    # return prediction accuracy of algorithm
    return pd.DataFrame({'Actual': ytest, 'Predicted': ypred})


xysplit = split(attributes, label)
classifydf = treeClassify(xysplit[0], xysplit[1], xysplit[2], xysplit[3], plot=True)

regressiondf = treeRegression(xysplit[0], xysplit[1], xysplit[2], xysplit[3], plot=True)

# print out accuracy of predicitons
print("\nAccuracy of Predictions for Classifier Tree:")
print(classifydf)

print("\nAccuracy of Predictions for Regression Tree:")
print(regressiondf)