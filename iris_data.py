import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data",
names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class"])

print(df.head(10))

df.hist(bins=20)

data_array=df.values
np.random.shuffle(data_array)
x = data_array[:80][:,0:4]
y = data_array[:80][:,4]

svc=SVC()
svc.fit(x,y)

X = data_array[-20:][:,0:4]
Y = data_array[-20:][:,4]

pred = svc.predict(X)
print(pred)
per=0
if print((pred==Y).all()):
    rate=100
else:
    for i in range(0,len(Y)):
        if pred[i]==Y[i]:
            per+=1


