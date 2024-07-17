import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

df=pd.read_csv("train.csv")
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']
X_cat = df[['Pclass', 'Sex',  'Embarked']]
X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
    
for col in X_cat.columns:
    X_cat.loc[:, col] = X_cat[col].fillna(X_cat[col].mode()[0])
for col in X_num.columns:
    X_num.loc[:, col] = X_num[col].fillna(X_num[col].median())
    
X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
X = pd.concat([X_cat_scaled, X_num], axis = 1)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
scaler = StandardScaler()
X_train.loc[:, X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test.loc[:, X_num.columns] = scaler.transform(X_test[X_num.columns])
    
def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

choice = ['Random Forest', 'SVC', 'Logistic Regression']

clf = prediction('Random Forest')
joblib.dump(clf, "model_random_forest.joblib")

clf = prediction('SVC')
joblib.dump(clf, "model_svc.joblib")

clf = prediction('Logistic Regression')
joblib.dump(clf, "model_logistic_regression.joblib")