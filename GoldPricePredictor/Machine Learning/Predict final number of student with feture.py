import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model
import pickle

#khandane filei ke kenar file .py ast
data=pd.read_csv("student-mat.csv",sep=";")

print(data.head())

#che sotoonhaii ro bekhanad(2D)
data=data[["G1", "G2","G3","studytime","failures","absences"]]

#che chizi ro pishbini konim???
predict="G3"

#hameye etelaat daneshgooha ro be joz G3 too table neshoon bede
x=np.array(data.drop([predict],1))

#G3 ro dar y neshoon bede
y=np.array(data[predict])

best=0
#in for baraye peyda kardane behtarin deghat ast ta oon ro save konim
for _ in range(1000):
    # x= list vijhegihaa , y= list javab ha
    # train va test mizarim
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.01)
    # linier model mizarim ki khat hadsie javab ro pishbini konad
    linear = linear_model.LinearRegression()
    #khat fit ro peyda kon
    linear.fit(x_train,y_train)
    #deghat khate fit ro peyda konim, in taghiri dar javab ijad nemkonad
    acc=linear.score(x_test,y_test)

    print(acc)

    #entekhab behtarin halat
    if acc>best:
        best=acc
        #dige hardafe train nemikone behtarin train ro dar(linear.fit)zakhire mikone
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)

print("Best:",best)
#package ro load mikonim  rb=readable
newModel=pickle.load(open("studentmodel.pickle","rb"))
#parametr B ya shib va co efficient baraye nemoodar khati linear regression
print("Coefficient:",newModel.coef_)
print("intercept:",newModel.intercept_)

#didane natayej nahaii(javabaye khorooji ra hads bezan va da result beriz
results=newModel.predict(x_test)
#baraye har data ye pishbini shode va daneshjooyan namayesh bede:
for x in range (len(results)): 
    print((results[x],x_test[x],y_test[x]))