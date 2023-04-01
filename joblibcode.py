import pandas as pd
import numpy as np
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt
data = pd.read_csv("train.csv")
df=pd.DataFrame(data)
data.dropna()
df.dropna()
"""sns.barplot(data=df,x="Credit_Score",y="Annual_Income")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Amount_invested_monthly")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Monthly_Balance")"""
'''plt.show()
sns.barplot(data=df,x="Credit_Score",y="Monthly_Inhand_Salary")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Num_Bank_Accounts")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Num_Credit_Card")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Interest_Rate")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Num_of_Loan")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Delay_from_due_date")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Num_of_Delayed_Payment")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Outstanding_Debt")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Credit_Utilization_Ratio")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Credit_History_Age")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Total_EMI_per_month")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Amount_invested_monthly")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Monthly_Balance")
plt.show()
sns.barplot(data=df,x="Credit_Score",y="Monthly_Balance")
plt.show()'''
data["Credit_Mix"] = data["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})
x = np.array(data[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
#print(data.isnull().sum())
y = np.array(data["Credit_Score"])
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.33,random_state=42)
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
model =RandomForestClassifier(max_depth=20)
model.fit(xtrain,ytrain)
dump(model, "model.joblib")
'''print("Accuracy of randomforest =",metrics.accuracy_score(ytest,model.predict(xtest)))
print("Accuracy of Decisiontree =",metrics.accuracy_score(ytest,model1.predict(xtest)))
print("Accuracy of logisticregession =",metrics.accuracy_score(ytest,model2.predict(xtest)))
print("Credit Score Prediction : ")
print('hello0')
a = float(input("Annual Income: "))
print('hello1')
b = float(input("Monthly Inhand Salary: "))
print('hello2')
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))
x=[[a, b, c, d, e, f, g, h, i, j, k, l]]
features = np.array(x, dtype=float)'''