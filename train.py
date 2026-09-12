import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#visualization importer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


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

# Susun Scikit-Learn Pipeline - randomforest
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


#Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)

labels = np.array([
    [f"True Negative (TN)\n{cm[0,0]}", f"False Positive (FP)\n{cm[0,1]}"],
    [f"False Negative (FN)\n{cm[1,0]}", f"True Positive (TP)\n{cm[1,1]}"]
])

# Tambahkan cbar_kws={'label': 'Jumlah Pasien'} di sini
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
            xticklabels=['Sehat', 'Berisiko'], 
            yticklabels=['Sehat', 'Berisiko'], 
            cbar_kws={'label': 'Jumlah Pasien'}, ax=axes[0, 0])

axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Prediksi Model')
axes[0, 0].set_ylabel('Kondisi Aktual')


# 2. ROC Curve (Kanan Atas -> axes[0, 1])
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.4f})', color='darkorange', lw=2)
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Tebakan Acak')
axes[0, 1].set_title('Receiver Operating Characteristic(ROC) Curve')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].legend(loc="lower right")

# Menambahkan kotak teks Akurasi secara terpisah di sudut kiri atas
axes[0, 1].text(0.05, 0.95, f'Akurasi Model = {acc:.4f}', 
                transform=axes[0, 1].transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

# 3. Feature Importance (Kiri Bawah -> axes[1, 0])
importances = pipeline.named_steps['classifier'].feature_importances_
feature_df = pd.DataFrame({'Fitur': X.columns, 'Kepentingan': importances}).sort_values(by='Kepentingan', ascending=False)

sns.barplot(x='Kepentingan', y='Fitur', data=feature_df, palette='viridis', hue='Fitur', legend=False, ax=axes[1, 0])
axes[1, 0].set_title('Feature Importance')
axes[1, 0].set_xlabel('Tingkat Pengaruh')
axes[1, 0].set_ylabel('')


# 4. Heatmap
num_cols = df.select_dtypes(include="number").columns
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".1f",
            cmap="coolwarm", center=0, linewidths=0.5, 
            annot_kws={"size": 8}, ax=axes[1, 1]) 
axes[1, 1].set_title("Heatmap")

axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
axes[1, 1].set_yticklabels(axes[1, 1].get_yticklabels(), rotation=0)

fig.suptitle(
    f"Dashboard Evaluasi Model Random Forest\n(Total Data: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)})", 
    fontsize=16, fontweight='bold'
)

# Merapikan grafik
plt.tight_layout(pad=4.0, h_pad=10.0, w_pad=5.0)
plt.subplots_adjust(top=0.82)
plt.show()