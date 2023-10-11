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
print('Accuracy score: ',accuracy_score(y_predict,y_test))
print('Precision_score: ',precision_score(y_predict,y_test))
print('Recall_score: ',recall_score(y_predict,y_test))
print('F1_score:',f1_score(y_predict,y_test))
#Visualize decision tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,8))
# from sklearn import tree
# print(tree.plot_tree(clf))


