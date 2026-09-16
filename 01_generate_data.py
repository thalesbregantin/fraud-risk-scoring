"""
Geração de dataset sintético de transações financeiras (Pix, TED, Boleto, Cartão)
para simular um problema de scoring de risco/fraude.

Abordagem: sorteamos primeiro a classe (fraude vs. legítima) na prevalência-alvo
e, condicionado à classe, sorteamos cada feature de uma distribuição diferente
(fraudes tendem a ter valores mais altos, dispositivos novos, contas recentes,
maior distância da localização habitual, etc). Isso é o equivalente, em dados
tabulares, ao que approaches como make_classification fazem: cria um sinal
real e controlável, com sobreposição entre as classes (não são perfeitamente
separáveis) — o que é o comportamento esperado em um problema de fraude real.

Dados 100% sintéticos, sem qualquer PII real. Servem apenas como base de estudo
para um projeto de portfólio de modelagem de risco/fraude.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 60_000
TAXA_FRAUDE_ALVO = 0.012

is_fraude = RNG.binomial(1, TAXA_FRAUDE_ALVO, size=N)
n_fraude = is_fraude.sum()
n_legit = N - n_fraude
idx_fraude = is_fraude == 1
idx_legit = ~idx_fraude

print(f"N={N} | fraudes={n_fraude} ({n_fraude/N*100:.2f}%) | legitimas={n_legit}")


def sample_por_classe(dist_legit, dist_fraude):
    """Aplica uma função de amostragem diferente para cada classe e concatena."""
    out = np.empty(N)
    out[idx_legit] = dist_legit(n_legit)
    out[idx_fraude] = dist_fraude(n_fraude)
    return out


df = pd.DataFrame({"id_transacao": np.arange(1, N + 1)})

# Canal: fraude concentra mais em Pix e Cartão (canais mais visados)
canal = np.empty(N, dtype=object)
canal[idx_legit] = RNG.choice(
    ["Pix", "TED", "Boleto", "Cartao"], size=n_legit, p=[0.50, 0.18, 0.17, 0.15]
)
canal[idx_fraude] = RNG.choice(
    ["Pix", "TED", "Boleto", "Cartao"], size=n_fraude, p=[0.45, 0.08, 0.07, 0.40]
)
df["canal"] = canal

# Valor da transação: fraudes tendem a valores um pouco mais altos (com bastante sobreposição)
df["valor_transacao"] = sample_por_classe(
    lambda n: RNG.lognormal(mean=4.1, sigma=1.0, size=n),
    lambda n: RNG.lognormal(mean=4.7, sigma=1.15, size=n),
).round(2)

# Hora da transação: fraude com maior probabilidade de ocorrer de madrugada
def horas_legit(n):
    return (RNG.normal(loc=14, scale=4.5, size=n).round().astype(int)) % 24

def horas_fraude(n):
    # mistura: 72% padrão diurno, 28% madrugada (1h-5h)
    mask_madrugada = RNG.random(n) < 0.28
    horas = (RNG.normal(loc=14, scale=5, size=n).round().astype(int)) % 24
    horas_madrugada = RNG.integers(1, 6, size=n)
    return np.where(mask_madrugada, horas_madrugada, horas)

df["hora_transacao"] = sample_por_classe(horas_legit, horas_fraude).astype(int)

# Dia da semana (pouco informativo — ruído de propósito)
df["dia_semana"] = RNG.integers(0, 7, size=N)

# Idade da conta em dias: fraude concentrada em contas mais novas (com sobreposição)
df["idade_conta_dias"] = sample_por_classe(
    lambda n: (RNG.exponential(scale=420, size=n) + 20).round().astype(int).clip(1, 4000),
    lambda n: (RNG.exponential(scale=140, size=n) + 1).round().astype(int).clip(1, 4000),
).astype(int)

# Nº de transações do usuário nas últimas 24h: fraude com maior velocidade
df["qtd_transacoes_24h"] = sample_por_classe(
    lambda n: RNG.poisson(lam=1.2, size=n),
    lambda n: RNG.poisson(lam=2.6, size=n),
).clip(0, 25).astype(int)

# Desvio do valor em relação à média histórica do usuário (z-score aproximado)
df["desvio_valor_media_historica"] = sample_por_classe(
    lambda n: RNG.normal(loc=0.0, scale=0.9, size=n),
    lambda n: RNG.normal(loc=1.3, scale=1.3, size=n),
)

# Distância (km) entre a localização da transação e a localização habitual
df["distancia_localizacao_habitual_km"] = sample_por_classe(
    lambda n: RNG.exponential(scale=10, size=n),
    lambda n: RNG.exponential(scale=42, size=n),
).round(1)

# Mudança de dispositivo (1 = transação feita em dispositivo nunca usado antes)
df["mudanca_dispositivo"] = sample_por_classe(
    lambda n: RNG.binomial(1, 0.05, size=n),
    lambda n: RNG.binomial(1, 0.34, size=n),
).astype(int)

# Score de reputação do dispositivo/IP (0-1, quanto menor mais suspeito)
df["score_reputacao_dispositivo"] = sample_por_classe(
    lambda n: RNG.beta(a=7, b=1.5, size=n),
    lambda n: RNG.beta(a=2.6, b=2.4, size=n),
)

df["is_fraude"] = is_fraude

# Embaralha as linhas (evita qualquer ordenação por classe)
df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
df["id_transacao"] = np.arange(1, N + 1)

print("Shape:", df.shape)
print("Taxa de fraude final:", df["is_fraude"].mean().round(4))
print(df["is_fraude"].value_counts())

df.to_csv("data_transacoes.csv", index=False)
print("Arquivo salvo: data_transacoes.csv")
