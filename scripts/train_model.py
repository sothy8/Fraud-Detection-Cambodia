import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def main():
    data_path = 'data/processed/features.csv'
    model_path = 'models/fraud_model.pkl'
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv(data_path)
    X = df[['amount_zscore','time_since_last','is_new_recipient','is_new_device','hour']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
        eval_metric='auc', n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = (model.predict_proba(X_test)[:,1] > 0.5).astype(int)
    print(classification_report(y_test, y_pred, zero_division=0))
    print('ROC-AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

    joblib.dump(model, model_path)
    print('✅ Model saved to', model_path)

if __name__ == '__main__':
    main()
