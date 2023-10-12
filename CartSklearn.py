import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#criterion = 'gini' => CART
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
y_test = np.array(y_test)
clf = DecisionTreeClassifier(criterion="gini",max_depth=11,min_samples_split=2).fit(X_train,y_train)
y_predict = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('CartTree')
acc = accuracy_score(y_predict,y_test)
precision_score = precision_score(y_predict,y_test)
recall_score = recall_score(y_predict,y_test)
f1_score = f1_score(y_predict,y_test)
print('Accuracy score: ',acc)
print('Precision_score: ',precision_score)
print('Recall_score: ',recall_score)
print('F1_score:',f1_score)
print('Ty le du doan sai cua CartTree')
print('Accuracy score: ',1-acc)
print('Precision_score: ',1-precision_score)
print('Recall_score: ',1-recall_score)
print('F1_score:',1-f1_score)
#Visualize decision tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,8))
# from sklearn import tree
# print(tree.plot_tree(clf))


