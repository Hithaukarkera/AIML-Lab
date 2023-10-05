from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

Iris = load_iris()
x = Iris.data
print(x)

y = Iris.target
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

knn_clf = KNeighborsClassifier(n_neighbors=50)
knn_clf.fit(x_train,y_train)
y_predict = knn_clf.predict(x_test)


accuracy = accuracy_score(y_test,y_predict)
print("Accuracy: ", accuracy)
print()
prediction = knn_clf.predict([[1.5,2.6,3.5,1.1]])
print("Prediction : ", prediction)
print()
print("Confusion Matrix : ", confusion_matrix(y_predict,y_test))

correct = 4
incorrect = 4

for i in range(len(y_test)):
    if y_predict[i] == y_test[i]:
        correct += 1
    else:
        incorrect += 1
        
print("Correct Prediction:",correct)
print("InCorrect Prediction:",incorrect)
print()