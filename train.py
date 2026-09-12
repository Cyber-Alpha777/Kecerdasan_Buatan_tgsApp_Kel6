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
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay



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
fig, axes = plt.subplots(4, 1, figsize=(12, 24)) 
fig.patch.set_facecolor('white') 
# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.array([
    [f"True Negative (TN)\n{cm[0,0]}", f"False Positive (FP)\n{cm[0,1]}"],
    [f"False Negative (FN)\n{cm[1,0]}", f"True Positive (TP)\n{cm[1,1]}"]
])
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
            xticklabels=['Sehat', 'Berisiko'], 
            yticklabels=['Sehat', 'Berisiko'], 
            cbar_kws={'label': 'Jumlah Pasien'}, ax=axes[0], square=True)
axes[0].set_title('Confusion Matrix (Random Forest)', pad=15)
axes[0].set_xlabel('Prediksi Model')
axes[0].set_ylabel('Kondisi Aktual')

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.4f})', color='darkorange', lw=2)
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Tebakan Acak')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")
axes[1].text(0.05, 0.95, f'Akurasi Model = {acc:.4f}', 
             transform=axes[1].transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))

# 3. Feature Importance
importances = pipeline.named_steps['classifier'].feature_importances_
feature_df = pd.DataFrame({'Fitur': X.columns, 'Kepentingan': importances}).sort_values(by='Kepentingan', ascending=False)
sns.barplot(x='Kepentingan', y='Fitur', data=feature_df, palette='viridis', hue='Fitur', legend=False, ax=axes[2])
axes[2].set_title('Feature Importance')
axes[2].set_xlabel('Tingkat Pengaruh')
axes[2].set_ylabel('')

# 4. Heatmap
num_cols = df.select_dtypes(include="number").columns
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".1f",
            cmap="coolwarm", center=0, linewidths=0.5, 
            annot_kws={"size": 9}, ax=axes[3]) 
axes[3].set_title("Correlation Heatmap")
axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45, ha='right')
axes[3].set_yticklabels(axes[3].get_yticklabels(), rotation=0)

fig.suptitle(
    f"Dashboard Evaluasi Model Random Forest\n(Total Data: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)})", 
    fontsize=16, fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=5.0)
plt.close() 

#Scrollbar
root = tk.Tk()
root.title("Dashboard Evaluasi Model")
root.geometry("1100x800") 
root.configure(bg='white') 
main_frame = tk.Frame(root, bg='white')
main_frame.pack(fill=tk.BOTH, expand=1)

# Canvas dengan background putih dan tanpa border
canvas = tk.Canvas(main_frame, bg='white', highlightthickness=0)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)
second_frame = tk.Frame(canvas, bg='white')
canvas_window = canvas.create_window((0,0), window=second_frame, anchor="nw")

def configure_canvas(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas_width = event.width
    frame_width = second_frame.winfo_reqwidth()
    if canvas_width > frame_width:
        canvas.coords(canvas_window, ((canvas_width - frame_width) / 2, 0))
    else:
        canvas.coords(canvas_window, (0, 0))
canvas.bind('<Configure>', configure_canvas)

# Render matplotlib ke Tkinter
canvas_tk = FigureCanvasTkAgg(fig, second_frame)
canvas_tk.draw()
canvas_tk.get_tk_widget().pack()

#scrollwheel inoput
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
root.bind_all("<MouseWheel>", _on_mousewheel)

root.mainloop()