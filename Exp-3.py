import pandas as dm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

dataset=dm.read_csv('D:/Coding/Programming/Application_of_ml/WineQT.csv')
print(dataset.head())
print(dataset.info())
print(dataset.describe().T)
df=dm.DataFrame(dataset)

df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# dataset.hist(bins=25,figsize=(10,10))
# plt.show()
correlation=dataset.corr()
print(correlation)

#null values
p=dataset.isnull().sum()
print(p)

#spliting dataset
X=df.drop(['quality','best quality'],axis=1)
y=df['best quality']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)

#Normalizing
Nor=MinMaxScaler()
Nor_fit=Nor.fit(X_train)
X_train=Nor_fit.transform(X_train)
X_test=Nor_fit.transform(X_test)
print(X_train)
#models
models=[SVC(kernel='rbf')]
for i in range(1):
    models[i].fit(X_train,y_train)
    print('Training Accuracy : ', metrics.roc_auc_score(y_train, models[i].predict(X_train)))
    print('Validation Accuracy : ', metrics.roc_auc_score(y_test, models[i].predict(X_test)))
   

y_train_pred = models[0].predict(X_train)
y_test_pred = models[0].predict(X_test)

#Evaluation
# ROC curve and AUC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
auc_train = roc_auc_score(y_train, y_train_pred)
auc_test = roc_auc_score(y_test, y_test_pred)


plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, label=f"Train ROC Curve (AUC = {auc_train:.2f})")
plt.plot(fpr_test, tpr_test, label=f"Test ROC Curve (AUC = {auc_test:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
