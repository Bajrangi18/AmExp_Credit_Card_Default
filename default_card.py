import pandas as pd

from google.colab import drive
drive.mount("/content/drive")

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

dataTest = pd.read_csv('/content/drive/My Drive/dataSets/test.csv')

dataTrain = pd.read_csv('/content/drive/My Drive/dataSets/train.csv')

dataTrain.drop(['customer_id','name','gender'],axis=1,inplace=True)
dataTrain['owns_car'] = dataTrain['owns_car'].replace(['N','Y'],value=[0,1])
dataTrain['owns_house'] = dataTrain['owns_house'].replace(['N','Y'],value=[0,1])
dataTrain['occupation_type'] = dataTrain['occupation_type'].replace(['Unknown'],value=0.0)
dataTrain['occupation_type'] = dataTrain['occupation_type'].replace(['Laborers', 'Core staff', 'Accountants', 'High skill tech staff'
 ,'Sales staff', 'Managers' ,'Drivers', 'Medicine staff', 'Cleaning staff',
 'HR staff' ,'Security staff', 'Cooking staff', 'Waiters/barmen staff',
 'Low-skill Laborers' ,'Private service staff', 'Secretaries',
 'Realty agents', 'IT staff'],value=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
#dataTrain.drop('occupation_type',axis=1,inplace=True)#comment this
#dataTrain.drop('owns_house',axis=1,inplace=True)
dataTrain.drop('owns_car',axis=1,inplace=True)
#dataTrain.drop('no_of_children',axis=1,inplace=True)
#dataTrain.drop('age',axis=1,inplace=True)#comment this
dataTrain = dataTrain.fillna(0)

X = dataTrain.values[:,:14]
Y = dataTrain['credit_card_default']
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.09, random_state = 80)
clf = DecisionTreeClassifier(criterion='entropy', splitter='best',max_features="auto",
                             random_state=100,min_impurity_decrease=0.0)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred)*100)


dataTest.drop(['customer_id','name','gender'],axis=1,inplace=True)
dataTest['owns_car'] = dataTest['owns_car'].replace(['N','Y'],value=[0,1])
dataTest['owns_house'] = dataTest['owns_house'].replace(['N','Y'],value=[0,1])
dataTest['occupation_type'] = dataTest['occupation_type'].replace(['Unknown'],value=0.0)
dataTest['occupation_type'] = dataTest['occupation_type'].replace(['Laborers', 'Core staff', 'Accountants', 'High skill tech staff'
 ,'Sales staff', 'Managers' ,'Drivers', 'Medicine staff', 'Cleaning staff',
 'HR staff' ,'Security staff', 'Cooking staff', 'Waiters/barmen staff',
 'Low-skill Laborers' ,'Private service staff', 'Secretaries',
 'Realty agents', 'IT staff'],value=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
dataTest.drop('occupation_type',axis=1,inplace=True)#comment this
dataTest.drop('owns_house',axis=1,inplace=True)
dataTest.drop('owns_car',axis=1,inplace=True)
dataTest = dataTest.fillna(0)


W = dataTest.values[:,:12]
W

w_pred = clf.predict(W)
