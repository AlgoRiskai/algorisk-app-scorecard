import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Generate mock application data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.randint(20000, 120000, n),
    'loan_amount': np.random.randint(1000, 30000, n),
    'default': np.random.binomial(1, 0.2, n)
})

# Step 2: WoE and IV Calculation
def bin_variable(df, var, target, bins=5):
    df[f'{var}_bin'] = pd.qcut(df[var], q=bins, duplicates='drop')
    grouped = df.groupby(f'{var}_bin', observed=True).agg({target: ['count', 'sum']})
    grouped.columns = ['total', 'bads']
    grouped['goods'] = grouped['total'] - grouped['bads']
    total_goods = grouped['goods'].sum()
    total_bads = grouped['bads'].sum()
    grouped['dist_good'] = grouped['goods'] / total_goods
    grouped['dist_bad'] = grouped['bads'] / total_bads
    grouped['WoE'] = np.log((grouped['dist_good'] + 0.0001) / (grouped['dist_bad'] + 0.0001))
    grouped['IV'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['WoE']
    woe_dict = grouped['WoE'].to_dict()
    iv = grouped['IV'].sum()
    return df[f'{var}_bin'].map(woe_dict), iv

# Step 3: Transform features with WoE
features = ['age', 'income', 'loan_amount']
iv_values = {}
for feature in features:
    df[f'{feature}_woe'], iv = bin_variable(df, feature, 'default')
    iv_values[feature] = iv

# Step 4: Prepare training data
woe_features = [f'{f}_woe' for f in features]
X = df[woe_features].fillna(0)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Step 6: Score scaling function
def scale_scores(probs, base_score=600, pdo=50):
    odds = (1 - probs) / probs
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(20)
    return offset + factor * np.log(odds)

scores = scale_scores(y_pred_prob)

# Step 7: Evaluate performance
auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Print IVs and coefficients
print("\nInformation Value (IV):")
for k, v in iv_values.items():
    print(f"{k}: {v:.4f}")

print("\nModel Coefficients:")
for name, coef in zip(woe_features, model.coef_[0]):
    print(f"{name}: {coef:.4f}")

print(f"\nAUC Score: {auc:.4f}")
