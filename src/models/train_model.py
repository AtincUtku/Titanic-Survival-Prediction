from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model():
    # Load the cleaned data
    train_data = pd.read_csv('../data/processed/train.csv')

    # Separate features and target
    X = train_data.drop(['PassengerId', 'Survived'], axis=1)
    y = train_data['Survived']

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    rf = RandomForestClassifier(n_estimators=35, max_depth=7, random_state=1)
    rf.fit(X_train, y_train)

    # Calculate the accuracy of the model on training data
    acc_train = rf.score(X_train, y_train)
    print(f'Training Accuracy: {acc_train*100:.2f}%')

    # Make predictions on the validation set
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)[:, 1]

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Calculate precision, recall, F1 score and AUC-ROC
    precision = precision_score(y_val, y_pred)
    print(f'Precision: {precision*100:.2f}%')

    recall = recall_score(y_val, y_pred)
    print(f'Recall: {recall*100:.2f}%')

    f1 = f1_score(y_val, y_pred)
    print(f'F1 Score: {f1*100:.2f}%')

    auc_roc = roc_auc_score(y_val, y_proba)
    print(f'AUC-ROC: {auc_roc*100:.2f}%')

    # Return the trained model
    return rf
