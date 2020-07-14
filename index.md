# LOAN PREDICTION USING MACHINE LEARNING

In finance, a loan is lending of the money from an individual or a group to an individual or an organisation etc.
The borrower requests for the loan and after he/she is approved of the loan, has to repay the sum along with the interest laid on the amount borrowed.
It is a hercules job to estimate if the borrower will repay the loan.hence it should be automated.

Here, we analyse and predict the loan approval using some machine learning algorithms.
 
**SOURCE:** The dataset for this project is retrieved from [kaggel](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)

**PROBLEM STATEMENT:** To predict which of the customer will repay the loan.

**ALGORITHM USED:**
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
Property_Area | Urban/Semi/Rural
Loan_Status | Y/N
 
**Data Cleaning:**

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

```python
loan_train.head()
```
```python
loan_train.info()
```
```python
loan_train.describe()
```

**Check for missing values**

We can use heat map to observe the missing values
It can be calculated by

```python
total_null = loan_train.isnull().sum().sort_values(ascending=False)
miss_data = pd.concat([total_null], axis=1, keys=['Total'])
miss_data.head(13)
```

