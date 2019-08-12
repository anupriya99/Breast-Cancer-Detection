
# Title : This Program detects breast cancer , based of a data

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv')
df.head(7)

# count the no of rows and columns
df.shape # result is 569 rows and 33 columns so 569 no of patients records and 33 column means 33 features

# count the no of empty values in each column
df.isna().sum()

#Drop the column with all missing values 
df=df.dropna(axis=1)

#Get the new count of the no of rows and column
df.shape

# get a count  of the number of malignant (M) or Benign (B) cells
df['diagnosis'].value_counts()

#visualize the count
sns.countplot(df['diagnosis'], label='count')

#Check the data types that which columns need to be encoded
df.dtypes

#Encode the categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

# doing above code the column diagnosis got encoded now the malignant (M) represent 1 and Benign (B) represent 0

df.head(2)

# Create a pair plot
sns.pairplot(df.iloc[:,1:5] , hue='diagnosis')

# Get the correlation of the columns
df.iloc[:,1:12].corr()

# To visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='.0%')

#split the dataset into independent dataset(X)[X will tell us the features to detect the cancer] and dependent  (Y) [y will tell us if the patient has cancer or not] data sets
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

# Split the dataset into 75% training data and 25% test data
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Scale the data (Feature Scaling)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Create a function for the models 

def models(X_train , Y_train):
  #Logistic Regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state=0)
  log.fit(X_train,Y_train)
  
  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
  tree.fit(X_train,Y_train)
  
  #Random Forest Classifier
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10,criterion='entropy',random_state=0)
  forest.fit(X_train,Y_train)
  
  #Print the model accuracy on the training data
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train,Y_train))
  print('[1]Decision Tree Classifier Training Accuracy:', tree.score(X_train,Y_train))
  print('[2]Random Forest Classifier Training Accuracy:', forest.score(X_train,Y_train))
  
  return log,tree,forest

# Checking all the models
model = models(X_train, Y_train)

# Test our model accuracy in Testing data using confusion matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,model[0].predict(X_test))
print(cm)

#true positive is 86 , true negative is 50 , false positive is 4 and false

TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Testing Accuracy = ', (TP+TN)/(TP+TN+FN+FP))

# so the below result is testing dataset accuracy is 95%

# Test our model accuracy in Testing data using confusion matrix 

from sklearn.metrics import confusion_matrix

for i in range(len(model)):
  print('Model ',i)
  cm = confusion_matrix(Y_test,model[i].predict(X_test))
  TP = cm[0][0]
  TN = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]
  print(cm)
  print('Testing Accuracy = ', (TP+TN)/(TP+TN+FN+FP))
  print()

#Alternative way to see metrics of the models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
  print('Model',i)
  print(classification_report(Y_test,model[i].predict(X_test)))
  print(accuracy_score(Y_test,model[i].predict(X_test)))
  print()

# Print the prediction of RandomForest Classifier Model
pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)