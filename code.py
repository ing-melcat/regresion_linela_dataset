import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

print("Versión de kagglehub:", kagglehub.__version__)

DATASET = "lalish99/covid19-mx"
path = kagglehub.dataset_download(DATASET)
print("\nDataset descargado en:", path)

csv_files = glob.glob(os.path.join(path, "*.csv"))

if not csv_files:
    raise FileNotFoundError("No se encontró ningún archivo CSV.")

selected_csv = None
for f in csv_files:
    name = os.path.basename(f).lower()
    if "general" in name or "mx" in name or "covid" in name:
        selected_csv = f
        break

if selected_csv is None:
    selected_csv = csv_files[0]

print("Usando archivo:", selected_csv)

df = pd.read_csv(selected_csv, low_memory=False)
df.columns = [c.strip().lower() for c in df.columns]

print("\nColumnas detectadas:")
print(df.columns.tolist())

if "fecha_def" not in df.columns:
    raise ValueError("No se encontró la columna 'fecha_def'.")

df["fecha_def"] = df["fecha_def"].astype(str).str.strip()
y = (df["fecha_def"] != "9999-99-99").astype(int)

features = [
    "sexo",
    "tipo_paciente",
    "intubado",
    "neumonia",
    "edad",
    "diabetes",
    "epoc",
    "asma",
    "inmusupr",
    "hipertension",
    "otra_con",
    "cardiovascular",
    "obesidad",
    "renal_cronica",
    "tabaquismo",
    "uci"
]

features = [col for col in features if col in df.columns]

if len(features) < 2:
    raise ValueError("No hay suficientes variables para entrenar el modelo.")

X = df[features].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([97, 98, 99, 999, 9999], np.nan)
X = X.fillna(X.median(numeric_only=True))

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

print("\nFeatures finales usadas:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=3000, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
intercepto = model.intercept_[0]

coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": model.coef_[0]
}).sort_values("Coeficiente", ascending=False)

print("\n================ RESULTADOS ================")
print("Score:", score)
print("Intercepto:", intercepto)
print("\nCoeficientes:")
print(coeficientes.to_string(index=False))

coef_plot = coeficientes.sort_values("Coeficiente", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(coef_plot["Variable"], coef_plot["Coeficiente"])
plt.xlabel("Coeficiente")
plt.ylabel("Variable")
plt.title("Coeficientes de la Regresión Logística")
plt.grid(True, axis="x")
plt.tight_layout()
plt.show()
