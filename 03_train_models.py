"""
Feature engineering + treinamento de modelos de scoring de risco/fraude.

Modelos:
  - Regressão Logística (baseline interpretável, class_weight='balanced')
  - XGBoost (modelo principal, scale_pos_weight para lidar com desbalanceamento)

Métricas: ROC-AUC, PR-AUC (average precision), matriz de confusão e
precision/recall em um ponto de operação de negócio (ex: capturar ~70% das
fraudes com o menor número possível de falsos positivos).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RNG_SEED = 42

df = pd.read_csv("data_transacoes.csv")

# --- Feature engineering ---
df = pd.get_dummies(df, columns=["canal"], prefix="canal")
df["transacao_noturna"] = df["hora_transacao"].between(1, 5).astype(int)
df["valor_x_velocidade"] = df["valor_transacao"] * (df["qtd_transacoes_24h"] + 1)
df["conta_nova"] = (df["idade_conta_dias"] < 30).astype(int)

feature_cols = [c for c in df.columns if c not in ("id_transacao", "is_fraude")]
X = df[feature_cols]
y = df["is_fraude"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
)

# --- Modelo 1: Regressão Logística (baseline) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG_SEED)
logreg.fit(X_train_scaled, y_train)
proba_logreg = logreg.predict_proba(X_test_scaled)[:, 1]

# --- Modelo 2: XGBoost ---
pos = y_train.sum()
neg = len(y_train) - pos
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=RNG_SEED,
)
xgb.fit(X_train, y_train)
proba_xgb = xgb.predict_proba(X_test)[:, 1]

# --- Métricas ---
def summarize(name, y_true, proba, threshold=0.5):
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return {
        "modelo": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdadeiros_positivos": int(tp),
    }

# threshold de negócio: buscar o menor threshold que capture >=70% das fraudes (recall alvo)
def threshold_para_recall_alvo(y_true, proba, recall_alvo=0.70):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve retorna thresholds com len = len(precisions)-1
    idx_candidatos = np.where(recalls[:-1] >= recall_alvo)[0]
    if len(idx_candidatos) == 0:
        return 0.5
    # entre os que atingem o recall alvo, pega o de maior threshold (mais preciso)
    melhor_idx = idx_candidatos[np.argmax(thresholds[idx_candidatos])]
    return float(thresholds[melhor_idx])

resultados = []
for nome, proba in [("Regressao_Logistica", proba_logreg), ("XGBoost", proba_xgb)]:
    resultados.append(summarize(f"{nome} (thr=0.5)", y_test, proba, threshold=0.5))
    thr_negocio = threshold_para_recall_alvo(y_test, proba, recall_alvo=0.70)
    resultados.append(summarize(f"{nome} (thr_negocio~recall70%)", y_test, proba, threshold=thr_negocio))

with open("metricas_modelos.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

for r in resultados:
    print(r)

# --- Gráfico: Curvas ROC ---
fig, ax = plt.subplots(figsize=(5.5, 5))
RocCurveDisplay.from_predictions(y_test, proba_logreg, name="Regressão Logística", ax=ax, color="#3B82F6")
RocCurveDisplay.from_predictions(y_test, proba_xgb, name="XGBoost", ax=ax, color="#EF4444")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
ax.set_title("Curva ROC — comparação de modelos")
plt.tight_layout()
plt.savefig("plot_5_curva_roc.png", dpi=150)
plt.close()

# --- Gráfico: Curvas Precision-Recall ---
fig, ax = plt.subplots(figsize=(5.5, 5))
PrecisionRecallDisplay.from_predictions(y_test, proba_logreg, name="Regressão Logística", ax=ax, color="#3B82F6")
PrecisionRecallDisplay.from_predictions(y_test, proba_xgb, name="XGBoost", ax=ax, color="#EF4444")
ax.set_title("Curva Precision-Recall (mais informativa p/ classes raras)")
plt.tight_layout()
plt.savefig("plot_6_curva_precision_recall.png", dpi=150)
plt.close()

# --- Gráfico: Importância de features (XGBoost) ---
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6.5, 6))
importances.tail(12).plot(kind="barh", ax=ax, color="#8B5CF6")
ax.set_title("Importância das features — XGBoost")
ax.set_xlabel("Importância (gain relativo)")
plt.tight_layout()
plt.savefig("plot_7_importancia_features.png", dpi=150)
plt.close()

# --- Gráfico: Matriz de confusão (XGBoost, threshold de negócio) ---
thr_negocio_xgb = threshold_para_recall_alvo(y_test, proba_xgb, recall_alvo=0.70)
preds_negocio = (proba_xgb >= thr_negocio_xgb).astype(int)
cm = confusion_matrix(y_test, preds_negocio)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Legítima", "Fraude"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Legítima", "Fraude"])
ax.set_xlabel("Predito"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusão — XGBoost\n(threshold={thr_negocio_xgb:.3f}, recall alvo 70%)")
plt.tight_layout()
plt.savefig("plot_8_matriz_confusao.png", dpi=150)
plt.close()

print("\nThreshold de negócio (XGBoost, recall alvo 70%):", round(thr_negocio_xgb, 4))
print("Gráficos e métricas salvos com sucesso.")
