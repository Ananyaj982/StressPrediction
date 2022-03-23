#KNN model 2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression



fields = ['Sub_ID','HeartRate', 'SpO2', 'Lab_val','Label']
url=r'D://sem6//capstone//syn1.csv'
dataset = pd.read_csv(url, usecols=fields, engine='python')
X=dataset[['Sub_ID','HeartRate','SpO2']]
X = X[:].values
#separate target values
y = dataset['Lab_val'].values
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

def plot():
    plt.figure(figsize=(8,6))

    plt.scatter(dataset['SpO2'], dataset['HeartRate'],
            color='green', label='input scale', alpha=0.5)
    #plt.scatter(df[:0], df[:1],color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)
    plt.title('SpO2 and HR content of the physiological dataset')
    plt.xlabel('SpO2')
    plt.ylabel('HR')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

def logistic_Regression(X_train, y_train,query):
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    # fit the model with data
    logreg.fit(X_train,y_train)
    y_pred=logreg.predict(query)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print("Precision:",metrics.precision_score(y_test, y_pred))
    #print("Recall:",metrics.recall_score(y_test, y_pred))
    #cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #print(cnf_matrix)
    prob_test = logreg.predict_proba(query)
    return [y_pred, prob_test]

def validate(y_test,pred_test):
  # Confusion Matrix
  print(metrics.confusion_matrix(y_test,pred_test))
  print('True target values: ',y_test[0:25])
  print('Predicted target values: ',pred_test[0:25])
  print()
  confusion = metrics.confusion_matrix(y_test,pred_test)
  TP = confusion[1,1]
  TN = confusion[0,0]
  FP = confusion[0,1]
  FN = confusion[1,0]
  # Metrics calclulation using confusion matrix
  print()
  # Classsification accuracy:- how often is the classifier correct
  print('Classification Accuracy:- ' , metrics.accuracy_score(y_test,pred_test))
  # Classification error/Misclassification rate:- how often is the classifier is incorrect
  print('Classification Error:- ' , 1-metrics.accuracy_score(y_test,pred_test))
  # Sensitivity :- when the actual value is positive , how often is the prediction correct?
  print('Sensitivity:- ' , metrics.recall_score(y_test,pred_test))
  # Specificity:- when the actual value is negative ,how often the prediction is the correct?
  print('Specificity:- ' , TN/float(TN+FP))
  # False positive rate:- when the actual value is negative ,how often the prediction is the incorrect?
  print('False positive rate:- ' , FP/float(TN+FP))
  # Precision:- when a positive value is predicted , how often is the prediction correct?
  print('Precision:- ' , metrics.precision_score(y_test,pred_test))

def pred(hr):
  pred_test = []
  count = 0
  #pred = knn(X_train,y_train)
  for i in hr:
        clf_query = [i]
        pred_test.append(logistic_Regression(X, y,clf_query))
        count += 1
        #print(count)
  #validate(y_test,pred_test)
  return pred_test
  #print('Accuracy measure for dataset:- ' , '{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
  #plot()
  #plt.show()


def main():
  #pred([[4,87,96]])
  return None



if __name__ == '__main__':
    main()
