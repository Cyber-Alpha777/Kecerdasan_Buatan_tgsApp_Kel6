import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,accuracy_score, confusion_matrix,roc_auc_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print('Membaca dataset heart.csv...')
df = pd.read_csv('heart.csv')

# Hapus baris duplikat agar evaluasi model objektif
df = df.drop_duplicates().reset_index(drop=True)
print(f'Jumlah data unik yang digunakan: {len(df)}')

# Pisahkan fitur dan target (target: 1 = Berisiko Jantung, 0 = Sehat)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Susun Scikit-Learn Pipeline
pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        (
            'classifier',
            LogisticRegression(max_iter=300, random_state=42),
        ),
    ]
)

print('Melatih model logistic regression...')
pipeline.fit(X_train, y_train)

# Evaluasi metrik
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("═══ Logistic Regression ═══")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Heart Disease Risk"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Simpan Pipeline jika memenuhi syarat >= 0.70
if auc >= 0.70 or acc >= 0.70:
  payload = {'pipeline': pipeline, 'features': list(X.columns)}
  joblib.dump(payload, 'model.joblib')
  print(
      'Target tercapai (>= 0.70). File "model.joblib" berhasil disimpan!'
  )
else:
  print('Gagal: Metrik belum mencapai 0.70.')


#Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    pipeline, X_test, y_test,
    display_labels=["Healthy", "Heart Disease Risk"],
    ax=ax, colorbar=False, cmap="Blues"
)
ax.set_title("Logistic Regression")
plt.tight_layout()
plt.show()
