import pandas as pd
import joblib

def make_predictions():
    # Load the cleaned test data
    test_data = pd.read_csv('../data/processed/test.csv')

    if test_data['Fare'].isnull().sum() > 0:
        # If 'Fare' column in test_data has NaN values,
        # Load training data to calculate the median of 'Fare'
        train_data = pd.read_csv('../data/processed/train.csv')
        median_fare = train_data['Fare'].median()

        # Fill NaN values in 'Fare' column in test_data with the median of 'Fare' from train_data
        test_data['Fare'].fillna(median_fare, inplace=True)


    # Load the saved model
    model = joblib.load('../models/random_forest.pkl')

    # Separate features
    X_test = test_data.drop(['PassengerId'], axis=1)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Prepare a submission file
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('../outputs/predictions.csv', index=False)
    print("Your submission was successfully saved!")

if __name__ == '__main__':
    make_predictions()
