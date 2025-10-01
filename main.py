from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt

sc_data = pd.DataFrame()
t_data = pd.read_csv("D:\Downloads\\train.csv")

X = t_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X = pd.get_dummies(X, dtype=int)
X = X.drop(['Sex_male'], axis=1).rename(columns={'Sex_female' : 'Sex'}).fillna({'Age' : X.Age.median()})
y = t_data.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
par = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
grid_search_clf = GridSearchCV(clf, par, cv=5)
grid_search_clf.fit(X_train, y_train)
best_clf = grid_search_clf.best_estimator_
y_pred = best_clf.predict(X_test)
pres = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f'Precision score:{pres}', f'Recall score:{rec}', sep='\n')
print(f'CV Score:{cross_val_score(best_clf, X_test, y_test, cv=5).mean()}')
