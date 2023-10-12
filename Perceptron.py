import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
#criterion = 'entropy' => CART
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
pla = Perceptron(fit_intercept=True,alpha=0.05).fit(X_train,y_train)
y_predict = pla.predict(X_test)
from sklearn.metrics import accuracy_score
print('Perceptron')
acc = accuracy_score(y_predict,y_test)
precision_score = precision_score(y_predict,y_test)
recall_score = recall_score(y_predict,y_test)
f1_score = f1_score(y_predict,y_test)
print('Accuracy score: ',acc)
print('Precision_score: ',precision_score)
print('Recall_score: ',recall_score)
print('F1_score:',f1_score)
print('Ty le du doan sai cua Perceptron')
print('Accuracy score: ',1-acc)
print('Precision_score: ',1-precision_score)
print('Recall_score: ',1-recall_score)
print('F1_score:',1-f1_score)