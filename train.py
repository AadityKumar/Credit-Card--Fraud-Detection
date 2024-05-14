from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pandas as pd
import joblib

models={
                "KNeighbors" : KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "AdaBoost Classifier":AdaBoostClassifier(),
                "Logistic Regression":LogisticRegression()
            }
params = {
    "KNeighbors": {
        'n_neighbors': range(1, 21),  # Explore a wider range of neighbors (1 to 20)
    },
    "Decision Tree": {
        'max_depth': range(2, 11),  # Add max_depth for tree complexity control
        'min_samples_split': range(2, 21),  # Add min_samples_split for avoiding overfitting
    },
    "Random Forest": {
        'max_depth': range(2, 11),  # Add max_depth for tree complexity control
        'min_samples_split': range(2, 21),  # Add min_samples_split for avoiding overfitting
        'n_estimators': range(64, 513, 64),  # Adjust range for larger forest sizes
    },
    "Gradient Boosting": {
        'learning_rate': [0.1, 0.05, 0.01, 0.005],  # Adjust learning rates for finer tuning
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],  # Adjust subsample range for stability
        'max_depth': range(2, 11),  # Add max_depth for tree complexity control
        'min_samples_split': range(2, 21),  # Add min_samples_split for avoiding overfitting
    },
    "AdaBoost Classifier": {
        'learning_rate': [0.1, 0.05, 0.01, 0.005],  # Adjust learning rates for finer tuning
        'n_estimators': range(64, 513, 64),  # Adjust range for larger ensemble sizes
    },
    "Logistic Regression": {
        'C': [1.0, 0.1, 0.01],  # Regularization parameter for controlling model complexity
        'solver': ['liblinear', 'lbfgs']  # Choose a solver algorithm for optimization
    },
}
model_folder_path="Model"

#Function to evaluate different machine learning models to get best model
def evaluate_model(X_train, Y_train, X_test, Y_test, models, param):
    report = {}
    for model_name, model in models.items():
        para = param[model_name]

        gs = GridSearchCV(model, param_grid=para, cv=3)
        gs.fit(X_train, Y_train)  # Train Model

        best_params = gs.best_params_
        model.set_params(**best_params)  # Update model with best parameters

        model.fit(X_train, Y_train)
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)

        train_model_score = r2_score(Y_train, Y_train_pred)
        test_model_score = r2_score(Y_test, Y_test_pred)
        print('--------', model_name, '--------')
        print('Train Score:', train_model_score)
        print('Test Score:', test_model_score)
        report[model_name] = test_model_score

    return report

def find_best_model(X_train,Y_train,X_test,Y_test):
    model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,param=params)
    best_model_score=max(sorted(model_report.values()))
    best_mod=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
    print('Best Model',best_mod)
    print('Best Model Score:',best_model_score)

    best_model=models[best_mod]
    best_model.fit(X_train,Y_train)
    if model_folder_path:
            model_save_path = f"{model_folder_path}/model.joblib"
            joblib.dump(best_model, model_save_path)

def create_data():
    # load data
    data = pd.read_csv('creditcard.csv')

    # separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    data = pd.concat([legit_sample, fraud], axis=0)

    # split data into training and testing sets
    X = data.drop(columns="Class", axis=1)
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    return X_train,X_test,y_train,y_test
if __name__ =="__main__":
    X_train,X_test,y_train,y_test = create_data()
    find_best_model(X_train,y_train,X_test,y_test) 