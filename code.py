import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay

# Descargar dataset
DATASET = "lalish99/covid19-mx"
path = kagglehub.dataset_download(DATASET)
print("Dataset descargado en:", path)

# Buscar CSV
csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

selected_csv = None
preferred_keywords = ["general", "covid", "clinical", "mx"]

for f in csv_files:
    name = os.path.basename(f).lower()
    if any(k in name for k in preferred_keywords):
        selected_csv = f
        break

if selected_csv is None:
    selected_csv = csv_files[0]

print("Usando archivo:", selected_csv)

# Leer CSV
df = pd.read_csv(selected_csv, low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]

# Detectar variable objetivo
possible_target_cols = ["date_died", "fecha_def", "died", "death"]
target_col = None

for c in possible_target_cols:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    raise ValueError("No se encontró la columna objetivo.")

# Crear variable binaria
if target_col in ["date_died", "fecha_def"]:
    df[target_col] = df[target_col].astype(str).str.strip()
    y = (df[target_col] != "9999-99-99").astype(int)
else:
    y = df[target_col].astype(str).str.lower().isin(["1", "true", "yes", "dead", "deceased"]).astype(int)

# Seleccionar variables predictoras
candidate_features = [
    "sex", "patient_type", "intubed", "pneumonia", "age",
    "pregnancy", "diabetes", "copd", "asthma", "inmsupr",
    "hypertension", "other_disease", "cardiovascular",
    "obesity", "renal_chronic", "tobacco", "icu"
]

features = [col for col in candidate_features if col in df.columns]
X = df[features].copy()

# Limpiar valores
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].replace([97, 98, 99, 999], np.nan)

X = X.fillna(X.median(numeric_only=True))

valid_idx = ~y.isna()
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Resultados
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

score = model.score(X_test_scaled, y_test)
acc = accuracy_score(y_test, y_pred)
intercepto = model.intercept_[0]

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_[0]
}).sort_values("Coeficiente", ascending=False)

auc = roc_auc_score(y_test, y_prob)

print("Score:", score)
print("Accuracy:", acc)
print("Intercepto:", intercepto)
print(coeficientes)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Regresión Logística")
plt.legend()
plt.grid(True)
plt.show()

# Coeficientes
coef_plot = coeficientes.sort_values("Coeficiente", ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(coef_plot["Variable"], coef_plot["Coeficiente"])
plt.xlabel("Coeficiente")
plt.ylabel("Variable")
plt.title("Coeficientes del modelo")
plt.grid(True, axis="x")
plt.show()

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de confusión")
plt.show()
