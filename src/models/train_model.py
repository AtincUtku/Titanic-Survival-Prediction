from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model():
    # Load the cleaned data
    train_data = pd.read_csv('../data/processed/train.csv')


    # Separate features and target
    X = train_data.drop(['PassengerId', 'Survived'], axis=1)
    y = train_data['Survived']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    rf = RandomForestClassifier(n_estimators=37, max_depth=8, random_state=1)
    rf.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = rf.predict(X_val)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Return the trained model
    return rf

if __name__ == '__main__':
    train_model()
