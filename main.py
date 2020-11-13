import os
import pandas as pd
import numpy as np
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression


# method for initializing .csv file
def initCSV(csvfile):
    df = pd.read_csv(csvfile)

    # change width so all columns can be seen
    desired_width = 320
    pd.set_option('display.width', desired_width)
    np.set_printoptions(linewidth=desired_width)
    pd.set_option('display.max_columns', 15)

    return df


# method to select target column from dataset
def targetCol(df):
    print("Choose target attribute from available columns below: ")
    # display columns
    for i in range(len(df.columns)):
        print(str(i+1) + ". " + df.columns[i])
    # select target column
    while True:
        try:
            ans = int(input("Enter the number of the target column above: "))
            if ans not in range(1, len(df.columns)+1):
                raise ValueError
        except ValueError:
            print("Error: please enter a number between 1 and " + str(len(df.columns)) + ".\n")
        else:
            break
    return df.columns[ans]

# method to select file from those already in the directory
def existingFile():
    # current directory
    path_to_watch = "."

    listing = []
    for file in os.listdir(path_to_watch):
        try:
            pd.read_csv(file)
            listing.append(file)
        except (ValueError, PermissionError):
            pass

    for i in range(len(listing)):
        print(str(i+1) + ". " + listing[i])

    while True:
        try:
            ans = int(input("Enter the number of the file above that you'd like to use: "))
            if ans not in range(1, len(listing)+1):
                raise ValueError
        except ValueError:
            print("Error: please enter a number between 1 and " + str(len(listing)) + ".\n")
        else:
            break

    return listing[ans-1]


# method to monitor the current directory for added/removed files so that the user can add their own dataset
def uploadFile():
    # current directory
    path_to_watch = "."

    # get all files in directory
    before = []
    for file in os.listdir(path_to_watch):
        before.append(file)

    print("Add your .csv file to the current directory: " + os.getcwd())

    # continue monitoring file uploads until one is chosen
    chosen = False
    while not chosen:
        # list for updated listing
        after = []
        for file in os.listdir(path_to_watch):
            after.append(file)

        # if a file in after[] was not in before[] then it
        for file in after:
            if file not in before:
                print("Added: " + file)
                # send to confirm() to see if user wants this file. returns boolean that dictates if loop stops or not
                chosen = confirm(file)
                # if chosen, save file in final variable
                if chosen:
                    final = file
                # else return to start
                else:
                    start()
        for file in before:
            if file not in after:
                print("Removed: " + file)
                before.remove(file)

        # set new before listing for next iteration
        before = after

    # after loop, final file has been chosen
    return final


# method to confirm file choice and validity
def confirm(f):

    try:
       pd.read_csv(f)
    except (ValueError, PermissionError):
        print("Error: chosen file is not a readable .csv. Please upload a different file.")
        return False

    print("Would you like to use the file " + f + " ?")
    while True:
        try:
            ans = input("Enter 'y' or 'n': ")
            if ans not in ('y', 'Y', 'n', 'N'):
                raise ValueError
        except ValueError:
            print("Error: please enter 'y' for yes or 'n' for no.\n")
        else:
            break

    if ans in ('y', 'Y'):
        return True
    elif ans in ('n', 'N'):
        return False


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


# function to split data into training and testing for tree. test size is 1/5 of the data
def split(df, col):
    # drop target label to define attributes
    attributes = df.drop(col, axis=1)

    # df of target column
    targets = df[col]

    return attributes, targets


# function to make a classifier decision tree from data
def treeClassify(x, y, file, plot):

    # split x and y into train and test groups
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)

    # start classifier tree
    clf = tree.DecisionTreeClassifier()
    # teach tree with the training set
    clf.fit(xtrain, ytrain)
    # predict y based on testing x after training
    ypred = clf.predict(xtest)

    # if plot is true, export as visualized tree
    if plot:
        dot_data = tree.export_graphviz(clf, out_file=None, feature_names=x.columns, class_names=y, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(file.replace('.csv', '') + "_Classifier")
        print("Exported classifier tree to pdf")
    # return prediction accuracy of algorithm (inverse transformed back to string target values) and percent accuracy
    return pd.DataFrame({'Actual': (ytest), 'Predicted': (ypred)}), accuracy_score(ytest, ypred)*100


# function to make a regression decision tree from data
def treeRegression(x, y, file, plot):
    # convert target label into integers for evaluation
    # ['allow' 'deny' 'drop' 'reset-both'] mapped to [0, 1, 2, 3]
    le = preprocessing.LabelEncoder()

    le.fit(y)

    try:
        # string representation of labels
        float(y[0])
    except ValueError:
        # integer representation of labels
        y = le.transform(y)

    # split x and y into train and test groups
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)

    reg = tree.DecisionTreeRegressor()      # initialize regression tree
    reg = reg.fit(xtrain, ytrain)                 # fit
    ypred = reg.predict(xtest)

    # if plot is true, export as visualized tree
    if plot:
        dot_data = tree.export_graphviz(reg, out_file=None, feature_names=x.columns, class_names=y, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render(file.replace('.csv', '') + "_Regression")
        print("Exported regression tree to pdf")
    # return prediction accuracy of algorithm and percent accuracy
    return pd.DataFrame({'Actual': ytest, 'Predicted': ypred}), mean_squared_error(ytest, ypred)


# starting function to drive other methods based on user choices
def start():
    # intro message
    print("This program performs supervised machine learning predictive analysis on a chosen dataset.\n")

    print("What dataset would you like to use?\n"
          "1. Select an existing dataset in the directory\n"
          "2. Upload a new dataset\n")

    while True:
        try:
            ans = int(input("Choose an option above by entering its corresponding number: "))
            if ans not in (1, 2):
                raise ValueError
        except ValueError:
            print("Error: please enter the number 1 or 2.\n")
        else:
            break

    # choose file to use
    if ans == 1:
        chosenFile = existingFile()
    elif ans == 2:
        chosenFile = uploadFile()

    # turn chosen .csv file into panda dataframe
    data = initCSV(chosenFile)
    # print dataset
    print("Original Dataset:")
    print(data)

    # choose target attribute from dataset
    target = targetCol(data)

    # print(data.describe())

    xysplit = split(data, target)

    # choose operation
    print("Would you like to perform tree classification or regression analysis on the dataset?\n"
          "1. Classification\n"
          "2. Regression\n")

    while True:
        try:
            ans = int(input("Choose an option above by entering its corresponding number: "))
            if ans not in (1, 2):
                raise ValueError
        except ValueError:
            print("Error: please enter the number 1, 2, or 3.\n")
        else:
            break

    if ans == 1:
        classifydf = treeClassify(xysplit[0], xysplit[1], chosenFile, plot=True)
        # print out accuracy of predictions, round percent accuracy to 2 decimals
        print("\nAccuracy of Predictions for Classifier Tree: " + str(round(classifydf[1], 2)) + "%")
        print(classifydf[0])
    elif ans == 2:
        regressiondf = treeRegression(xysplit[0], xysplit[1], chosenFile, plot=True)
        print("\nRoot Mean Square Error for Regression Tree: " + str(round(regressiondf[1], 5)))
        print(regressiondf[0])

    # choose operation
    print("What dimensionality reduction technique do you want to perform?\n"
          "1. Missing Value Ratio\n"
          "2. High Correlation Filter\n")

    while True:
        try:
            ans = int(input("Choose an option above by entering its corresponding number: "))
            if ans not in (1, 2):
                raise ValueError
        except ValueError:
            print("Error: please enter the number 1, 2, or 3.\n")
        else:
            break

    if ans == 1:
        newData = missing(data)
    elif ans == 2:
        # remove high correlation and set as new dataframe
        newData = correlation(data)

    # check if changes were made
    if newData.equals(data):
        print("Data remains unchanged.")
    else:
        # option to save new data
        while True:
            try:
                ans = input("Would you like to save your new dataset as a new .csv file?\nEnter 'y' or 'n': ")
                if ans not in ('y', 'Y', 'n', 'N'):
                    raise ValueError
            except ValueError:
                print("Error: please enter 'y' for yes or 'n' for no.\n")
            else:
                break

        if ans in ('y', 'Y'):
            # save to new .csv
            save(newData)

    while True:
        try:
            ans = input("Would you like to operate on another .csv file?\nEnter 'y' or 'n': ")
            if ans not in ('y', 'Y', 'n', 'N'):
                raise ValueError
        except ValueError:
            print("Error: please enter 'y' for yes or 'n' for no.\n")
        else:
            break

    if ans in ('y', 'Y'):
        # restart program
        start()
    elif ans in ('n', 'N'):
        # end program
        print("Bye!")
        sys.exit()


start()
