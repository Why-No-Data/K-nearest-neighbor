import numpy as np
from sklearn import preprocessing , cross_validation, neighbors
import pandas as pd
import matplotlib.pyplot as plt


accuracies=[]
for i in range(1000):#training model n times

    df=pd.read_csv('breast-cancer-wisconsin.csv')
    df.replace('?',-99999,inplace=True)#-99999 for making outlier
    df.drop(['id'],1,inplace=True)


    X=np.array(df.drop(['class'],1))#features
    Y=np.array(df['class'])#labels or class

    X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(X,Y,test_size=0.2)

    clf=neighbors.KNeighborsClassifier(n_jobs=-1)#parrallel jobs
    clf.fit(X_train,Y_train)

    accuracy=clf.score(X_test,Y_test)
        #print(accuracy)

       ## example_measure=np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1],[8,10,7,8,9,10,3,2,1]])#never seen this data before
        #example_measure=example_measure.reshape(len(example_measure),-1) #if u are not using np.array[[---]]

        ##prediction=clf.predict(example_measure)
        #print(prediction)
    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))
