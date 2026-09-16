"""
Análise exploratória do dataset sintético de transações.
Gera gráficos usados no README/portfólio.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

df = pd.read_csv("data_transacoes.csv")

# 1) Balanceamento de classes
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["is_fraude"].value_counts().sort_index()
labels = ["Legítima", "Fraude"]
colors = ["#3B82F6", "#EF4444"]
bars = ax.bar(labels, counts.values, color=colors)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
ax.set_title(f"Distribuição de classes (taxa de fraude = {df['is_fraude'].mean()*100:.2f}%)")
ax.set_ylabel("Nº de transações")
plt.tight_layout()
plt.savefig("plot_1_distribuicao_classes.png", dpi=150)
plt.close()

# 2) Distribuição do valor da transação por classe (log scale)
fig, ax = plt.subplots(figsize=(6, 4))
for cls, color, label in [(0, "#3B82F6", "Legítima"), (1, "#EF4444", "Fraude")]:
    subset = df.loc[df["is_fraude"] == cls, "valor_transacao"]
    ax.hist(subset.clip(upper=subset.quantile(0.99)), bins=40, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel("Valor da transação (R$)")
ax.set_ylabel("Densidade")
ax.set_title("Distribuição do valor por classe")
ax.legend()
plt.tight_layout()
plt.savefig("plot_2_valor_por_classe.png", dpi=150)
plt.close()

# 3) Taxa de fraude por canal
fig, ax = plt.subplots(figsize=(5.5, 4))
taxa_canal = df.groupby("canal")["is_fraude"].mean().sort_values(ascending=False) * 100
bars = ax.bar(taxa_canal.index, taxa_canal.values, color="#F59E0B")
for b, v in zip(bars, taxa_canal.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}%", ha="center", va="bottom")
ax.set_ylabel("Taxa de fraude (%)")
ax.set_title("Taxa de fraude por canal")
plt.tight_layout()
plt.savefig("plot_3_taxa_por_canal.png", dpi=150)
plt.close()

# 4) Correlação entre features numéricas e o alvo
num_cols = [
    "valor_transacao", "hora_transacao", "dia_semana", "idade_conta_dias",
    "qtd_transacoes_24h", "desvio_valor_media_historica",
    "distancia_localizacao_habitual_km", "mudanca_dispositivo",
    "score_reputacao_dispositivo", "is_fraude",
]
corr = df[num_cols].corr(numeric_only=True)["is_fraude"].drop("is_fraude").sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#EF4444" if v < 0 else "#10B981" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.set_title("Correlação das features com is_fraude")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("plot_4_correlacao_features.png", dpi=150)
plt.close()

print("EDA concluída. Gráficos salvos:")
print(" - plot_1_distribuicao_classes.png")
print(" - plot_2_valor_por_classe.png")
print(" - plot_3_taxa_por_canal.png")
print(" - plot_4_correlacao_features.png")
print()
print(corr)
