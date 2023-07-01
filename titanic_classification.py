from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
titanic_train=pd.read_csv("C:/Users/Admin/OneDrive/Desktop/train.csv")
titanic_test=pd.read_csv("C:/Users/Admin/OneDrive/Desktop/test.csv")
titanic_train.head()    #top 5 records
titanic_train.shape   #total no. of records & columns
print(titanic_train['Survived'].value_counts())    #no. of counts of different type  of values stored
print(titanic_train['Survived'].value_counts().keys())    #to get only keys ,i.e., only the distinct no. not the actual count of keys
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Survived'].value_counts().keys()),list(titanic_train['Survived'].value_counts()),color=["red","green"])
plt.title("Survived")
plt.show()
print(titanic_train['Pclass'].value_counts())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Pclass'].value_counts().keys()),list(titanic_train['Pclass'].value_counts()),color=["orange","blue","black"])
plt.title("Passesnger Class")
plt.show()
print(titanic_train['Sex'].value_counts())
plt.figure(figsize=(5,5))
plt.bar(list(titanic_train['Sex'].value_counts().keys()),list(titanic_train['Sex'].value_counts()),color=["red","blue"])
plt.title("Sex")
plt.show()
plt.figure(figsize=(5,7))
plt.hist(titanic_train['Age'],color=["purple"])
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Age Group")
plt.show()
titanic_train["Survived"].isnull()  #if the column is having any null value then the output will be true otherwise the output will be false
print(sum(titanic_train['Survived'].isnull()))
titanic_train['Age'].isnull()  #if the column is having any null value then the output will be true otherwise the output will be false
print(sum(titanic_train['Age'].isnull()))
titanic_train=titanic_train.dropna() #dropna() is used to drop all the null values in the dataset. Here,after dropping all the null values the dataset is stored in titanic_train variable.
print(sum(titanic_train['Survived'].isnull()))  #here, we're cross-checking that the null valiues are still exixting or not.
print(sum(titanic_train['Age'].isnull()))   #here, we're cross-checking that the null valiues are still exixting or not.
x_train=titanic_train[["Age"]]
y_train=titanic_train[["Survived"]]
dtc=DecisionTreeClassifier()  #dtc is sthe model created from decision tree classifier
dtc.fit(x_train,y_train)   #it is used to fit parameters like class_weight,criterion etc. in the model. 
print(sum(titanic_test['Age'].isnull()))
titanic_test=titanic_test.dropna()
print(sum(titanic_test['Age'].isnull()))
x_test=titanic_test[['Age']]
y_pred=dtc.predict(x_test)  #predict function is used to predict the output of the data stored in the x_test dataset.
print(y_pred)  #if the output is 1 then the passenger is survived & if the value is 0 then the passenger isn't survived.


