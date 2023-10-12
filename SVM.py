# perceptron 
# tree id3 , cart 
# svm
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
df = pd.read_csv('.vscode/surveylungcancer.csv')
#min_sample_split so mau toi thieu de phan chia (Moi mot ,Node phai co toi thieu min_sample_split)
print(df.shape)
lb = preprocessing.LabelEncoder()
data = df.apply(lb.fit_transform)
dt_Train,dt_Test = train_test_split(data,test_size=0.3,shuffle=True)
X_train = dt_Train.iloc[:, :15] 
y_train = dt_Train.iloc[:, 15] 
X_test = dt_Test.iloc[:, :15] 
y_test = dt_Test.iloc[:, 15]
print('Ma tran du lieu X')
print(data)
print("Nhan Y")
print(y_train)
print("Tap du lieu train")
print(dt_Train)
print("Tap du lieu test")
print(dt_Test)
# svr = make_pipeline(StandardScaler(),SVC(gamma="auto",class_weight="balanced"))
svr = SVC(kernel="linear",C=1e2)
svr.fit(X_train, y_train)
y_predict = svr.predict(X_test) 
acc = accuracy_score(y_predict,y_test)
precision_score = precision_score(y_predict,y_test)
recall_score = recall_score(y_predict,y_test)
f1_score = f1_score(y_predict,y_test)
print("SVM")
print('Accuracy score: ',acc)
print('Precision_score: ',precision_score)
print('Recall_score: ',recall_score)
print('F1_score:',f1_score)
print('Ty le du doan sai')
print('Accuracy score: ',1-acc)
print('Precision_score: ',1-precision_score)
print('Recall_score: ',1-recall_score)
print('F1_score:',1-f1_score)
#Độ đo của SVM là tốt nhất