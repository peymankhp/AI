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

#in do list ro be do list dige taghsim mikonim baraye train kardan va test kardan(bakhshe trai va test nabayad hamo bebinand)

x_train, x_test , y_train , y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

print(x_test)

