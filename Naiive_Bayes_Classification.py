import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def readFile():

    dataset = pd.read_csv('C:\\Users\\datasets\\Buy_Book.csv')

    for column in dataset.columns:
        if dataset [column].dtype == type(object): 
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset [column])


    X = dataset.iloc[0: 0:4].values 
    Y = dataset.iloc[0:, -1].values
    return X, Y


def split_data(x,y):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    return X_train, X_test,y_train,y_test


def model_train(X_train,Y_train,X_test,Y_test):

    X_train1 = np.reshape(X_train, (-1,1))
    Y_train1 = np.reshape(Y_train, (-1,1))

    X_test1 = np.reshape(X_test, (-1,1))
    Y_test1 = np.reshape(Y_test, (-1,1))

    classifier =  GaussianNB()
    classifier.fit(X_train,Y_train)

    y_pred = classifier.predict(x_test)

    accuracy = accuracy_score (Y_test,y_pred)

    cm = confusion_matrix(Y_test,y_pred)

    tran_age = np.array([[27,0,0,0]])

    buy_bookp = classifier.predict(tran_age)

    return classifier


def visualize_result(X_train1,y_train1,X_test1,y_test1, classifier):

    pass


def main():

    X, Y = readFile()
    X_train, X_test, y_train, y_test = split_data(X,Y)
    classifier = model_train (X_train, y_train,X_test,X_test) 
    visualize_result(X_train,y_train,X_test,y_test, classifier)

if __name__ == "__main__":
    main()
    