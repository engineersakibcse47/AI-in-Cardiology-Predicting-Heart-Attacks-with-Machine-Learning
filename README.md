# AI in Cardiology: Predicting Heart Attacks with Machine Learning

## Introduction
### What is Heart Attack?
Discusses Myocardial infarction, commonly known as a heart attack, explaining how it occurs due to occlusion of vessels by cholesterol and fat, leading to blood flow blockage and potential fatality.

## Exploratory Data Analysis (EDA)
This section likely includes statistical analysis, data visualization, and examination of data distribution to gain insights into the dataset used for predicting heart attacks.

## Preparation for Modelling
Data preparation steps, such as cleaning, normalization, and splitting, are carried out to ensure the dataset is ready for modeling.

## Modelling
Multiple machine learning models are implemented and evaluated:

## Logistic Regression Algorithm

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {}".format(accuracy))
```

## Cross-validation

```
scores = cross_val_score(log_reg, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())
```
## Hyperparameter Optimization with GridSearchCV
```
log_reg_new = LogisticRegression()
parameters = {"penalty":["l1","l2"], "solver" : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
log_reg_grid = GridSearchCV(log_reg_new, param_grid = parameters)
log_reg_grid.fit(X_train, y_train)
log_reg_new2 = LogisticRegression(penalty = "l1", solver = "saga")
log_reg_new2.fit(X_train, y_train)
y_pred = log_reg_new2.predict(X_test)
print("The test accuracy score of Logistic Regression After hyper-parameter tuning is: {}".format(accuracy_score(y_test, y_pred)))
```

## Decision Tree Algorithm
```
dec_tree = DecisionTreeClassifier(random_state = 5)
dec_tree.fit(X_train, y_train)
y_pred = dec_tree.predict(X_test)
print("The test accuracy score of Decision Tree is:", accuracy_score(y_test, y_pred))

scores = cross_val_score(dec_tree, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())
```

## Support Vector Machine Algorithm

```
svc_model = SVC(random_state = 5)
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
print("The test accuracy score of SVM is:", accuracy_score(y_test, y_pred))
scores = cross_val_score(svc_model, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())
```

## Random Forest Algorithm
```
random_forest = RandomForestClassifier(random_state = 5)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print("The test accuracy score of Random Forest is", accuracy_score(y_test, y_pred))
scores = cross_val_score(random_forest, X_test, y_test, cv = 10)
print("Cross-Validation Accuracy Scores", scores.mean())
```
## Hyperparameter Optimization with GridSearchCV for Random Forest
```
random_forest_new = RandomForestClassifier(random_state = 5)
parameters = {"n_estimators" : [50, 100, 150, 200], 
              "criterion" : ["gini", "entropy"], 
              'max_features': ['auto', 'sqrt', 'log2'], 
              'bootstrap': [True, False]}

random_forest_grid = GridSearchCV(random_forest_new, param_grid = parameters)
random_forest_grid.fit(X_train, y_train)
print("Best Parameters:", random_forest_grid.best_params_)

random_forest_new2 = RandomForestClassifier(bootstrap = True, criterion = "entropy", max_features = "auto", n_estimators = 200, random_state = 5)

from sklearn.ensemble import RandomForestClassifier

random_forest_new2 = RandomForestClassifier(
    bootstrap=True, 
    criterion="entropy", 
    max_features=None,  # Changed from 'auto' to None
    n_estimators=200, 
    random_state=5
)

random_forest_new2.fit(X_train, y_train)

y_pred = random_forest_new2.predict(X_test)

print("The test accuracy score of Random Forest after hyper-parameter tuning is:", accuracy_score(y_test, y_pred))
```

## Conclusion

The activities we carried out within the scope of the project are as follows:

Within the scope of the project, we first made the data set ready for Exploratory Data Analysis(EDA)
I performed Exploratory Data Analysis(EDA).
I analyzed numerical and categorical variables within the scope of univariate analysis by using Distplot and Pie Chart graphics.
I made the data set ready for the model. In this context, we struggled with missing and outlier values.
I used four different algorithms in the model phase.
I got 87% accuracy and 88% AUC with the Logistic Regression model.
I got 83% accuracy and 85% AUC with the Decision Tree Model.
I got 83% accuracy and 89% AUC with the Support Vector Classifier Model.
And I got 90.3% accuracy and 94% AUC with the Random Forest Classifier Model.
All these model outputs are evaluated, we prefer the model we created with the Random Forest Algorithm, which gives the best results.



