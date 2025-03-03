import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

#data ro load mikonim
data=pd.read_csv("car.data")
# print(data.head())

#data ha ro tabdil mikonim be adaad ba method fit transport
le=preprocessing.LabelEncoder()

buying=le.fit_transform(list(data["buying"]))
maint=le.fit_transform(list(data["maint"]))
doors=le.fit_transform(list(data["doors"]))
persons=le.fit_transform(list(data["persons"]))
lug_boot=le.fit_transform(list(data["lug_boot"]))
safety=le.fit_transform(list(data["safety"]))
cls=le.fit_transform(list(data["class"]))

#do ta list dorost mikonim ke migim in khosoosiati ke in class dare dar in daste gharar migirand
x=list(zip(buying, maint, doors, persons, lug_boot,safety)) #features
#lable ha:
y=list(cls)

#in do list ro be do list dige taghsim mikonim baraye train kardan va test kardan(bakhshe train va test nabayad hamo bebinand)
#Model_selection: is a method for setting a blueprint to analyze data and then using it to measure new data -->
# Selecting a proper model allows you to generate accurate results when making a prediction.
#train_test_split: is a function in Sklearn model selection for splitting data arrays into two subsets:
# for training data and for testing data. With this function, you don't need to divide the dataset manually.
#By default, Sklearn train_test_split will make random partitions for the two subsets. However, you can also specify a random state for the operation.
x_train, x_test , y_train , y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

print(x_test)

#KNN nazdiktarin hamsayegi ro peyda mikonad
model= KNeighborsClassifier(n_neighbors=7)
#train kardan model:
model.fit(x_train,y_train)

#tashkhis deghat model:
acc=model.score(x_test,y_test)

print(acc)

#hanuz dataye jadid nadidim va miaiim inja ijad mikonim
predicted= model.predict(x_test)
clas=["unacc","acc","good","vgood"]

for x in range(len(predicted)):
    #moghayese claspredicted x ba y_test
    print('pred data: ', clas[predicted[x]],"   real data: ", clas[y_test[x]])
    #item haii ke da KNN nazdiktarin hastan ro estekhraj mikonim: 3= sta hamsayeh , True= Return konad = bale
    #baraye debug kardan mohem ast
    n=model.kneighbors([x_test[x]],3,True)
    # print("n=  ",n)
