import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
    print("Your predictions was successfully saved!")

    # Load the gender_submission file
    gender_submission = pd.read_csv('../data/raw/gender_submission.csv')

    # Extract the true survival values
    true_values = gender_submission['Survived']

    assert len(predictions) == len(true_values), 'Mismatch in length between predictions and true_values'
    assert all(test_data['PassengerId'] == gender_submission['PassengerId']), 'Mismatch in order of PassengerIds between test_data and gender_submission'


    # Calculate metrics
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions)
    recall = recall_score(true_values, predictions)
    f1 = f1_score(true_values, predictions)
    roc_auc = roc_auc_score(true_values, model.predict_proba(X_test)[:, 1])

    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1 Score: {f1*100:.2f}%')
    print(f'ROC AUC: {roc_auc*100:.2f}%')


if __name__ == '__main__':
    make_predictions()
