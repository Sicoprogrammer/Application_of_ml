import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

data = pd.read_csv("D:/Coding/Programming/Application_of_ml/creditcard.csv")
#print(data.head())
#print(data.shape)
#print(data.describe())

fraud = data[data['Class']==1]
valid=data[data['Class']==0]
outlier=len(fraud)/float(len(valid))
#print(outlier)
#print('fraud cases:{}'.format(len(data[data['Class']==1])))
#print('valid transactions:{}'.format(len(data[data["Class"]==0])))
#highly imbalanced dataset
#print(fraud.Amount.describe())
#print(valid.Amount.describe())
#plotting confusion matrix for feature selection
correlation=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(correlation, vmax=.8, square=True)
#plt.show()
X=data.drop(['Class'],axis=1)
Y=data["Class"]
xData=X.values
yData=Y.values
print(X.shape)
print(Y.shape)
xtrain,xtest,ytrain,ytest=train_test_split(xData,yData,test_size=0.3,random_state=42)
random=RandomForestClassifier()
random.fit(xtrain,ytrain)
ypred=random.predict(xtest)

n_outliers = len(fraud) 
n_errors = (ypred != ytest).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(ytest, ypred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(ytest, ypred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(ytest, ypred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(ytest, ypred) 
print("The F1-Score is {}".format(f1)) 
  
MCC = matthews_corrcoef(ytest, ypred) 
print("The Matthews correlation coefficient is{}".format(MCC)) 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(ytest, ypred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 