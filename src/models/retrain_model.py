import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

def retrain_model():
    # Load the cleaned data
    train_data = pd.read_csv('../data/processed/train.csv')

    # Separate features and target
    X = train_data.drop(['PassengerId', 'Survived'], axis=1)
    y = train_data['Survived']

    # Define the parameter values that should be searched
    n_estimators_range = list(range(1, 100))
    max_depth_range = list(range(1, 10))

    # Create a parameter grid: map the parameter names to the values that should be searched
    param_grid = dict(n_estimators=n_estimators_range, max_depth=max_depth_range)

    # Instantiate the model
    rf = RandomForestClassifier()

    # Instantiate the grid
    grid = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy')

    # Fit the grid with data
    grid.fit(X, y)

    # View the complete results
    grid.cv_results_

    # Examine the best model
    print(grid.best_score_)
    print(grid.best_params_)

    # Save the new model, replacing the old one
    joblib.dump(grid.best_estimator_, '../models/random_forest.pkl')
    print("Model retrained and saved!")

if __name__ == '__main__':
    retrain_model()
