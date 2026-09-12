import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
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
            RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
        ),
    ]
)

print('Melatih model Random Forest...')
pipeline.fit(X_train, y_train)

# Evaluasi metrik
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print('--- Hasil Evaluasi ---')
print(f'Akurasi : {acc:.4f}')
print(f'ROC-AUC : {auc:.4f}')

# Simpan Pipeline jika memenuhi syarat >= 0.70
if auc >= 0.70 or acc >= 0.70:
  payload = {'pipeline': pipeline, 'features': list(X.columns)}
  joblib.dump(payload, 'model.joblib')
  print(
      'Target tercapai (>= 0.70). File "model.joblib" berhasil disimpan!'
  )
else:
  print('Gagal: Metrik belum mencapai 0.70.')