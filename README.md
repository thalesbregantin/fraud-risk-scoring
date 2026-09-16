# Modelo de Scoring de Risco/Fraude em Transações Financeiras

Projeto de estudo/portfólio: construção de um motor de scoring de risco para
identificar transações potencialmente fraudulentas (Pix, TED, Boleto e Cartão),
do desenho dos dados até a avaliação de modelos — no mesmo espírito do que um
time de Prevenção a Fraude/Risco faz em produção.

## 1. Contexto e objetivo

Fraude em meios de pagamento é um problema clássico de classificação com
classes extremamente desbalanceadas (normalmente <2% das transações são
fraude) e alto custo assimétrico: deixar passar uma fraude custa caro, mas
bloquear demais transações legítimas gera fricção e perda de receita. O
objetivo aqui foi simular esse cenário e construir um score de risco capaz de
priorizar transações suspeitas para triagem (manual ou automática via agente
de IA), maximizando a captura de fraude a um custo controlado de falsos
positivos.

**Importante — transparência sobre os dados:** não tenho acesso a dados reais
de transações (por óbvias razões de privacidade/sigilo bancário). Os dados
usados aqui são **100% sintéticos**, gerados programaticamente (`01_generate_data.py`)
com uma lógica de geração condicionada à classe: dado que uma transação é
fraude ou não, cada feature é sorteada de uma distribuição diferente, com
sobreposição proposital entre as classes (não são perfeitamente separáveis).
O objetivo não é "provar" um modelo pronto para produção, e sim demonstrar o
processo de ponta a ponta: EDA, feature engineering, modelagem, avaliação com
métricas adequadas a classes raras e interpretação de resultados.

## 2. Dados

- 60.000 transações simuladas, taxa de fraude de **1,2%**.
- Canais: Pix, TED, Boleto, Cartão.
- Features: valor da transação, horário, dia da semana, idade da conta,
  quantidade de transações nas últimas 24h (velocidade), desvio do valor em
  relação à média histórica do usuário, distância da localização habitual,
  troca de dispositivo e score de reputação do dispositivo/IP.

![Distribuição de classes](plot_1_distribuicao_classes.png)

## 3. Principais achados da EDA

- Isoladamente, cada sinal de risco tem correlação linear fraca com a fraude
  (ex.: `mudanca_dispositivo` ≈ 0,13) — o padrão de fraude só fica claro
  quando várias variáveis se combinam. Isso reforça a necessidade de um
  modelo multivariado em vez de regras simples de corte único.
- O sinal mais forte isolado foi a **distância da localização habitual**
  (corr. 0,32), seguido por `score_reputacao_dispositivo` (-0,26).
- Fraude está concentrada em contas mais novas, dispositivos nunca usados
  antes e transações de madrugada — consistente com o padrão de conta
  tomada/fraude de identidade.

![Taxa de fraude por canal](plot_3_taxa_por_canal.png)
![Correlação das features](plot_4_correlacao_features.png)

## 4. Modelagem

Dois modelos, ambos tratando o desbalanceamento de classes:

| Modelo | Técnica de desbalanceamento |
|---|---|
| Regressão Logística (baseline interpretável) | `class_weight="balanced"` |
| XGBoost (modelo principal) | `scale_pos_weight` |

Split estratificado 75/25 treino/teste.

## 5. Resultados

| Modelo | ROC-AUC | PR-AUC (avg. precision) |
|---|---|---|
| Regressão Logística | 0,989 | 0,823 |
| XGBoost | 0,987 | 0,805 |

Em problemas de classes raras, **PR-AUC é a métrica mais informativa** que
ROC-AUC (que pode parecer "otimista" mesmo com muitos falsos positivos). Os
dois modelos ficaram próximos — sinal de que, neste dataset, a relação entre
features e fraude é majoritariamente aditiva. Eu esperaria uma vantagem maior
do XGBoost sobre modelos lineares em dados reais, onde as interações entre
variáveis costumam ser mais complexas e não-lineares.

![Curva ROC](plot_5_curva_roc.png)
![Curva Precision-Recall](plot_6_curva_precision_recall.png)

### Ponto de operação de negócio

Threshold fixo em 0,5 raramente é a escolha certa em fraude — o ponto de
corte deve refletir o apetite a risco do negócio. Simulei um cenário de
**meta de recall de 70%** (capturar 7 em cada 10 fraudes):

| Modelo | Threshold | Precisão | Recall | Falsos positivos (em ~15 mil transações) |
|---|---|---|---|---|
| Regressão Logística | 0,989 | 85,7% | 70,0% | 21 |
| XGBoost | 0,924 | 83,4% | 70,0% | 25 |

Ou seja: para capturar 70% das fraudes, seria necessário revisar (ou
bloquear/step-up de autenticação) um volume pequeno e administrável de
transações legítimas — um trade-off realista para times de triagem.

![Importância das features](plot_7_importancia_features.png)
![Matriz de confusão](plot_8_matriz_confusao.png)

## 6. Próximos passos (se fosse para produção)

- Validar com dados reais e monitorar *drift* de padrão de fraude ao longo do
  tempo (fraudadores adaptam comportamento rapidamente).
- Explorar agentes de IA/LLMs para triagem qualitativa dos casos sinalizados
  pelo score (ex.: analisar contexto textual de disputas, PDFs de
  faturamento, histórico de atendimento) — combinando scoring quantitativo
  com agentes para casos ambíguos.
- Calibração de probabilidade (Platt scaling / isotonic) se o score for usado
  para decisões de valor esperado (ex.: custo de bloqueio vs. custo de
  fraude).
- Testes A/B do ponto de corte antes de mudanças em produção.

## 7. Stack

Python, pandas, scikit-learn, XGBoost, matplotlib.

## Arquivos

- `01_generate_data.py` — geração do dataset sintético (gera `data_transacoes.csv` localmente; o CSV não é versionado aqui por ser um artefato grande e 100% reproduzível a partir do script)
- `02_eda.py` — análise exploratória
- `03_train_models.py` — feature engineering, treino e avaliação dos modelos
- `metricas_modelos.json` — métricas detalhadas
- `plot_*.png` — gráficos gerados

## Como rodar

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
python 01_generate_data.py   # gera data_transacoes.csv
python 02_eda.py             # gera plots 1-4
python 03_train_models.py    # treina os modelos e gera plots 5-8 + metricas_modelos.json
```
