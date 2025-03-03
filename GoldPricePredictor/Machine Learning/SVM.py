import sklearn
from sklearn import datasets
from sklearn import  svm
from sklearn import  metrics

#dataset default sklearn baraye cancer
cancer= datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

#data haye target va train x,y ro joda mikonim
x=cancer.data
y=cancer.target
# print(x,y)

#in do list ro be do list dige taghsim mikonim baraye train kardan va test kardan(bakhshe train va test nabayad hamo bebinand)
#Model_selection: is a method for setting a blueprint to analyze data and then using it to measure new data -->
# Selecting a proper model allows you to generate accurate results when making a prediction.
#train_test_split: is a function in Sklearn model selection for splitting data arrays into two subsets:
# for training data and for testing data. With this function, you don't need to divide the dataset manually.
#By default, Sklearn train_test_split will make random partitions for the two subsets. However, you can also specify a random state for the operation.
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)
#test size=0.2  ---> 80 darsad be train va 20 darsad be test

#5 ta khoone aval test ro neshun bede
# print(x_train[:5],y_train[:5])

#sakhte classifier, SVC= Support Vector Classifier, kernel = functioni ast ke ejaze mide behtarin model ro besazim
#c = soft margine , ta che mizan cheshm pooshi konad
clf=svm.SVC(kernel="linear",C=20)

#fit kardan clf, y_train= label ha
clf.fit(x_train,y_train)
#baraye shenasaii mizane deghat predict mikonim, yani data taze midim bebinim chand marde halaje
y_pred=clf.predict(x_test)
#sehat javab ro baramoon misanje, miad predict ro ba javabe vagheii moghayase mikonad
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)





