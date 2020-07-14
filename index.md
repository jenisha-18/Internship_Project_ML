# LOAN PREDICTION USING MACHINE LEARNING

In finance, *loan* is lending the money to an individual or an organisation etc by any individual or group or organisation.
The borrower requests for the loan and after he/she is approved  and granted the loan amount,the client has to repay the sum along with the interest laid on the amount borrowed in regular basis after a certain amount of time.
It is a hercules job to estimate if the borrower will repay the loan or not .Hence it need to be automated.

Here, we analyse and predict the loan approval using some simple machine learning algorithms.
 
**SOURCE:** The dataset is downloaded from the website [kaggel](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset).

**PROBLEM STATEMENT:** To predict which of the clients will repay the loan.

**ALGORITHMS USED:**

    1.Logistic Regression
    
    2.Decision Tree
    
    3.Random Forest
    
 **DESCRIPTION OF THE DATASET:**

COLUMNS | DESCRIPTION
------- | -----------
Loan_ID | Unique loan ID of the client
Gender| Male or Female
Married | Yes/No
Dependents | No. of people depending on the customer
Education | Graduate or undergraduate
Self_Employed | Yes/No
ApplicantIncome | Income of the client
CoapplicantIncome | Income of the coapplicant
LoanAmount | Amount in Thousands
Loan_Amount_Terms | Term of the loan in months
Credit_Histort | Repayment of previous loan
Property_Area | Urban/Semiurban/Rural
Loan_Status | Y/N
 
## Data Cleaning:

Firstly, Import the required libraries


```python
#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Read the CSV files

```python
#read the csv files

loan_train=pd.read_csv(r'C:\Users\Veena\Downloads\train1.csv')
loan_test=pd.read_csv(r'C:\Users\Veena\Downloads\test1.csv')
```


Observe the dataset information

![head](https://github.com/jenisha-18/Internship_Project_ML/blob/images/info1.png)


![info](https://github.com/jenisha-18/Internship_Project_ML/blob/images/info2.png)


![col](https://github.com/jenisha-18/Internship_Project_ML/blob/images/info3.png)


### Check for missing values before applying any algorithm

We can use *heat map* to observe the missing values

![heatmap](https://github.com/jenisha-18/Internship_Project_ML/blob/images/heatmap1.png)

We can see that there are few datas missing from the dataset. We need to fill them before using any algorithm for analysis.


Number of missing data from the dataset can be calculated by:

```python
total_null = loan_train.isnull().sum().sort_values(ascending=False)
miss_data = pd.concat([total_null], axis=1, keys=['Total'])
miss_data.head(13)
```

![missing value](https://github.com/jenisha-18/Internship_Project_ML/blob/images/mvalue1.png)


**Fill up the missing datas**

```python
loan_train['Credit_History'] = loan_train['Credit_History'].fillna(loan_train['Credit_History'].dropna().mode().values[0] )
loan_train['Self_Employed'] = loan_train['Self_Employed'].fillna(loan_train['Self_Employed'].dropna().mode().values[0] )
loan_train['Gender'] = loan_train['Gender'].fillna(loan_train['Gender'].dropna().mode().values[0] )
loan_train['Dependents'] = loan_train['Dependents'].fillna(loan_train['Dependents'].dropna().mode().values[0] )
loan_train['Loan_Amount_Term'] = loan_train['Loan_Amount_Term'].fillna(loan_train['Loan_Amount_Term'].dropna().mode().values[0] )
loan_train['LoanAmount'] = loan_train['LoanAmount'].fillna(loan_train['LoanAmount'].dropna().median() )
loan_train['Married'] = loan_train['Married'].fillna(loan_train['Married'].dropna().mode().values[0] )
```
 
 We have used *mode* to fill the missing values. But for the loanAmount it is better to use *median* as it causes less error when compared to mode in this case.


Cross check for missing datas if any

```python
total_null = loan_train.isnull().sum().sort_values(ascending=False)
miss_data = pd.concat([total_null], axis=1, keys=['Total'])
miss_data.head(13)
```

![missing value](https://github.com/jenisha-18/Internship_Project_ML/blob/images/mvalue2.png)

No missing datas found.


## Analyse the data using graphs

**Let's use count plot to analyse the dataset**

![cp](https://raw.githubusercontent.com/jenisha-18/Internship_Project_ML/images/cp1.png)

Here, We can observe that those who aren't self employed apply for loans mostly.


![cp](https://github.com/jenisha-18/Internship_Project_ML/blob/images/cp2.png)

Males are most likely to get approved for loan than Females


![cp](https://github.com/jenisha-18/Internship_Project_ML/blob/images/cp3.png)

Loan applicants who are married are most likely of getting approval for loan.


![cp](https://github.com/jenisha-18/Internship_Project_ML/blob/images/cp4.png)

Here, We can observe that applicants who live in Semiurban areas obtain more loans followed by Urban and Rural.


![cp](https://github.com/jenisha-18/Internship_Project_ML/blob/images/cp5.png)

Those who have history of returning the loans are most likely of getting approval for loan.


![cp](https://github.com/jenisha-18/Internship_Project_ML/blob/images/cp6.png)

Loan Applications are mostly 360 cyclic loan term.


**Analysing the data using FacetGrid**


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg1.png)

Male Graduate has the highest income when compared to others.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg2.png)

Here, We can see that Males have higher income than that of Females. Males who are married have higher income than that of unmarried ones.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg3.png)

Males with no dependents tend to have high income and income reduces as the dependents increases.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg4.png)

Individual who is unmarried with no dependents have higher income. And also, those who are married and with no dependents have greater income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg5.png)

Applicant who is married and graduate has greater income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg6.png)

Those who have good credit history have high income regardless of marital status.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg7.png)

Here we can see that individual who is not self employed but graduated has higher income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg8.png)

Not a self employed with no dependents has greater income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg9.png)

A graduate with no dependent has high income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg10.png)

A graduate with good Credit history has higher income.


![fg](https://github.com/jenisha-18/Internship_Project_ML/blob/images/fg11.png)

Individual with no dependents tend to have more income.


### Observe the dataset for data cleaning

![obj](https://github.com/jenisha-18/Internship_Project_ML/blob/images/obj1.png)

We can see that datatypes of few columns are object. It is better to change them to numeric datatypes


```python
conv_dtype = {'Male': 1, 'Female': 2, 'Yes': 1, 'No': 2, 'Graduate': 1, 'Not Graduate': 2, 'Urban': 3, 'Semiurban': 2,'Rural': 1, 'Y': 1, 'N': 0, '3+': 3}

#Apply to each and every element in the dataframe

loan_train = loan_train.applymap(lambda s: conv_dtype.get(s) if s in conv_dtype else s)
loan_test = loan_test.applymap(lambda s: conv_dtype.get(s) if s in conv_dtype else s)

```

Check the information of dataset again
![obj](https://github.com/jenisha-18/Internship_Project_ML/blob/images/obj2.png)

Here, we can observe that all the fields except for Loan_ID and Dependents has been converted from object Dtype.
We can drop Loan_ID from the dataframe.

```python

#drop the loan id
loan_train.drop('Loan_ID', axis = 1, inplace = True)

#Convert the type of dependents
Dependents_Train = pd.to_numeric(loan_train.Dependents)
Dependents_Test = pd.to_numeric(loan_test.Dependents)

#Drop the object type Dependents from the dataframe
loan_train.drop(['Dependents'], axis = 1, inplace = True)
loan_test.drop(['Dependents'], axis = 1, inplace = True)

#Concatanate the numeric dtype Dependents to dataframe
loan_train = pd.concat([loan_train, Dependents_Train], axis = 1)
loan_test = pd.concat([loan_test, Dependents_Test], axis = 1)

```

Checking the datatype of columns of the dataset

![obj](https://github.com/jenisha-18/Internship_Project_ML/blob/images/obj3.png)

Finally,
all the datatypes are in numeric datatype and unnecessary columns are removed

Now we can proceed towards training phase

## TRAINING PHASE

define X and y for training

```python
#define X and y
X = loan_train.drop('Loan_Status', axis = 1)
y = loan_train['Loan_Status']
```

```python

from sklearn.model_selection import train_test_split

#Spliting the train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

```
We have taken test_size = 0.33


## Logistic Regression

Import the LogisticRegression

```python
#import LogisticRegression
from sklearn.linear_model import LogisticRegression
```


Fit the model

```python
#fit the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
```

Predict

```python
ypred = lr.predict(X_test)
```

Check the accuracy using classsification_report, confusion_matrix and accuracy_score

```python
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,ypred))
print(accuracy_score(y_test,ypred))
print(classification_report(y_test,ypred))

```



![lr](https://github.com/jenisha-18/Internship_Project_ML/blob/images/lr.png)

Here,we got **0.83** accuracy_score in Logistic Regression.


## Decision Tree

Import the DecisionTreeClassifier
```python
from sklearn.tree import DecisionTreeClassifier
```


Fit the model

```python
#fit the model
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
```

Predict

```python
ypred_dtree=dtree.predict(X_test)
```

Check the accuracy using classsification_report, confusion_matrix and accuracy_score

```python

print(confusion_matrix(y_test,ypred_dtree))
print(accuracy_score(y_test,ypred_dtree))
print(classification_report(y_test,ypred_dtree))

```



![dt](https://github.com/jenisha-18/Internship_Project_ML/blob/images/dt.png)

Here,we got **0.75** accuracy_score in Decision Tree.

## Random Forest

Import the RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
```


Fit the model

```python
#fit the model
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
```

Predict

```python
rfc_pred=rfc.predict(X_test)

```

Check the accuracy using classsification_report, confusion_matrix and accuracy_score

```python

print(confusion_matrix(y_test,rfc_pred))
print(accuracy_score(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

```



![rf](https://github.com/jenisha-18/Internship_Project_ML/blob/images/rf.png)

Here,we got **0.81** accuracy_score in Random Forest.


## CONCLUSION

Here, We can see that **LogisticRegression** performs better based on accuracy_score; followed by **RandomForestClassifier** and **DecisonTreeClassifier**
