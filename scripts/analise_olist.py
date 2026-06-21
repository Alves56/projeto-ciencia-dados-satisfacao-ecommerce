# -*- coding: utf-8 -*-
"""
==============================================================================
PROJETO PRATICO DE CIENCIA DE DADOS  -  TEMA: E-COMMERCE (OLIST)
Foco: SATISFACAO DO CLIENTE - o que faz uma avaliacao ser boa ou ruim?
==============================================================================
Dados publicos: Brazilian E-Commerce Public Dataset by Olist (~100 mil pedidos).
6 tabelas integradas: pedidos, avaliacoes, itens, pagamentos, produtos, clientes.

Conceitos de Ciencia de Dados (9 de 9):
  1. Coleta | 2. Limpeza/Integracao/Transformacao | 3. Estatistica Descritiva
  4. KPIs | 5. Visualizacao | 6. Feature Engineering
  7. Modelagem (Classificacao + Regressao) | 8. ML (K-Means + Apriori)
  9. Metricas de Avaliacao
==============================================================================
"""
import os, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    r2_score, mean_absolute_error, silhouette_score)
from mlxtend.frequent_patterns import apriori, association_rules

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "savefig.transparent": True})
FIG = "figuras"; os.makedirs(FIG, exist_ok=True)

def secao(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ==========================================================================
# CONCEITO 1 - COLETA DE DADOS
# ==========================================================================
secao("CONCEITO 1 - COLETA DE DADOS")
pedidos = pd.read_csv("data/olist_orders_dataset.csv")
aval = pd.read_csv("data/olist_order_reviews_dataset.csv")
itens = pd.read_csv("data/olist_order_items_dataset.csv")
pagto = pd.read_csv("data/olist_order_payments_dataset.csv")
produtos = pd.read_csv("data/olist_products_dataset.csv")
clientes = pd.read_csv("data/olist_customers_dataset.csv")
traducao = pd.read_csv("data/product_category_name_translation.csv")
for nome, df in [("pedidos", pedidos), ("avaliacoes", aval), ("itens", itens),
                 ("pagamentos", pagto), ("produtos", produtos), ("clientes", clientes)]:
    print(f"  {nome:12s}: {df.shape[0]:>6} linhas x {df.shape[1]} colunas")

# ==========================================================================
# CONCEITO 2 - LIMPEZA, INTEGRACAO E TRANSFORMACAO
# ==========================================================================
secao("CONCEITO 2 - LIMPEZA, INTEGRACAO E TRANSFORMACAO")

# Transformacao: converter datas para o tipo data
cols_data = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
for c in cols_data:
    pedidos[c] = pd.to_datetime(pedidos[c], errors="coerce")

# Limpeza: manter apenas pedidos ENTREGUES com datas validas
ped = pedidos[(pedidos["order_status"] == "delivered") & pedidos["order_delivered_customer_date"].notna()].copy()
print(f"Pedidos entregues com data valida: {len(ped)} (de {len(pedidos)})")

# Avaliacoes: 1 nota por pedido (media quando ha mais de uma)
nota = aval.groupby("order_id")["review_score"].mean().round().reset_index()

# Itens agregados por pedido: preco, frete, qtd e 1 produto (para a categoria)
it = itens.groupby("order_id").agg(
    preco_total=("price", "sum"), frete_total=("freight_value", "sum"),
    n_itens=("order_item_id", "count"), product_id=("product_id", "first")).reset_index()

# Pagamentos agregados por pedido
pg = pagto.groupby("order_id").agg(
    valor_pago=("payment_value", "sum"), parcelas=("payment_installments", "max"),
    tipo_pagamento=("payment_type", "first")).reset_index()

# Produtos -> categoria em ingles (mais legivel)
prod = produtos.merge(traducao, on="product_category_name", how="left")
prod["categoria"] = prod["product_category_name_english"].fillna(prod["product_category_name"]).fillna("desconhecida")
prod = prod[["product_id", "categoria"]]

# INTEGRACAO: juntar tudo numa unica tabela por pedido
df = (ped.merge(nota, on="order_id")
         .merge(it, on="order_id")
         .merge(pg, on="order_id")
         .merge(prod, on="product_id", how="left")
         .merge(clientes[["customer_id", "customer_state", "customer_unique_id"]], on="customer_id"))
print(f"Tabela integrada (1 linha por pedido): {df.shape[0]} pedidos x {df.shape[1]} colunas")

# ==========================================================================
# CONCEITO 6 - FEATURE ENGINEERING
# ==========================================================================
secao("CONCEITO 6 - FEATURE ENGINEERING")
df["tempo_entrega_dias"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["atraso_dias"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
df["atrasou"] = (df["atraso_dias"] > 0).astype(int)
df["satisfeito"] = (df["review_score"] >= 4).astype(int)          # ALVO da classificacao
df["parcelado"] = (df["parcelas"] > 1).astype(int)
regioes = {"Norte": ["AC","AP","AM","PA","RO","RR","TO"],
           "Nordeste": ["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
           "Centro-Oeste": ["DF","GO","MT","MS"],
           "Sudeste": ["ES","MG","RJ","SP"], "Sul": ["PR","RS","SC"]}
estado_para_regiao = {uf: reg for reg, ufs in regioes.items() for uf in ufs}
df["regiao"] = df["customer_state"].map(estado_para_regiao).fillna("Outro")
# limpeza final: remover linhas sem tempo de entrega valido
df = df.dropna(subset=["tempo_entrega_dias", "atraso_dias", "preco_total"]).reset_index(drop=True)
df = df[df["tempo_entrega_dias"] >= 0]
print("Variaveis criadas: tempo_entrega_dias, atraso_dias, atrasou, satisfeito, parcelado, regiao")
print(f"Tabela final: {df.shape[0]} pedidos")

# ==========================================================================
# CONCEITO 3 - ESTATISTICA DESCRITIVA
# ==========================================================================
secao("CONCEITO 3 - ESTATISTICA DESCRITIVA")
desc = df[["review_score", "tempo_entrega_dias", "frete_total", "preco_total"]].describe().round(2)
print(desc)
print(f"\nNota -> media {df.review_score.mean():.2f} | mediana {df.review_score.median():.0f} | "
      f"moda {df.review_score.mode()[0]:.0f} | desvio {df.review_score.std():.2f}")

# ==========================================================================
# CONCEITO 4 - INDICADORES (KPIs)
# ==========================================================================
secao("CONCEITO 4 - INDICADORES (KPIs)")
n = len(df)
atrasados = df[df.atrasou == 1]
no_prazo = df[df.atrasou == 0]
kpis = {
    "01. Pedidos analisados": n,
    "02. Faturamento total (R$)": df.preco_total.sum(),
    "03. Ticket medio (R$)": df.preco_total.mean(),
    "04. Nota media (1-5)": df.review_score.mean(),
    "05. Clientes satisfeitos (%)": 100 * df.satisfeito.mean(),
    "06. Pedidos atrasados (%)": 100 * df.atrasou.mean(),
    "07. Tempo medio de entrega (dias)": df.tempo_entrega_dias.mean(),
    "08. Frete medio (R$)": df.frete_total.mean(),
    "09. Itens por pedido (media)": df.n_itens.mean(),
    "10. Parcelas medias": df.parcelas.mean(),
    "11. Pedidos parcelados (%)": 100 * df.parcelado.mean(),
    "12. Nota media - ATRASADOS": atrasados.review_score.mean(),
    "13. Nota media - NO PRAZO": no_prazo.review_score.mean(),
    "14. Satisfeitos entre atrasados (%)": 100 * atrasados.satisfeito.mean(),
    "15. Satisfeitos no prazo (%)": 100 * no_prazo.satisfeito.mean(),
    "16. Categorias distintas": df.categoria.nunique(),
    "17. Categoria mais vendida": df.categoria.mode()[0],
    "18. Estados atendidos": df.customer_state.nunique(),
    "19. Regiao com mais pedidos": df.regiao.mode()[0],
    "20. Correlacao atraso x nota": df.atraso_dias.corr(df.review_score),
    "21. % pago no cartao de credito": 100 * (df.tipo_pagamento == "credit_card").mean(),
    "22. Maior tempo de entrega (dias)": df.tempo_entrega_dias.max(),
    "23. Desvio padrao da nota": df.review_score.std(),
    "24. Nota media - regiao Sudeste": df[df.regiao == "Sudeste"].review_score.mean(),
}
for k, v in kpis.items():
    print(f"  {k:38s}: {v:.2f}" if isinstance(v, float) else f"  {k:38s}: {v}")

# ==========================================================================
# CONCEITO 5 - VISUALIZACAO
# ==========================================================================
secao("CONCEITO 5 - VISUALIZACAO (figuras em figuras/)")
AZUL, VERM, VERDE = "#4C72B0", "#C44E52", "#55A868"

plt.figure(figsize=(7,5))  # G1 distribuicao das notas
df.review_score.value_counts().sort_index().plot.bar(color=AZUL)
plt.title("Distribuição das notas de avaliação"); plt.xlabel("Nota"); plt.ylabel("Pedidos")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(f"{FIG}/g01_notas.png"); plt.close()

plt.figure(figsize=(7,5))  # G2 nota media: atrasou vs nao (INSIGHT principal)
df.groupby("atrasou")["review_score"].mean().plot.bar(color=[VERDE, VERM])
plt.title("Nota média: no prazo vs. atrasado"); plt.ylabel("Nota média")
plt.xticks([0,1], ["No prazo", "Atrasado"], rotation=0); plt.tight_layout()
plt.savefig(f"{FIG}/g02_atraso_nota.png"); plt.close()

plt.figure(figsize=(7,5))  # G3 tempo de entrega por satisfacao
sns.boxplot(data=df, x="satisfeito", y="tempo_entrega_dias", palette=[VERM, VERDE])
plt.title("Tempo de entrega vs. satisfação"); plt.xticks([0,1], ["Insatisfeito", "Satisfeito"])
plt.ylabel("Tempo de entrega (dias)"); plt.ylim(0, 60); plt.tight_layout()
plt.savefig(f"{FIG}/g03_tempo_satisfacao.png"); plt.close()

plt.figure(figsize=(8,6))  # G4 correlacao
num = ["review_score","tempo_entrega_dias","atraso_dias","frete_total","preco_total","n_itens","parcelas"]
sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlação"); plt.tight_layout(); plt.savefig(f"{FIG}/g04_correlacao.png"); plt.close()

plt.figure(figsize=(8,5))  # G5 top categorias
df.categoria.value_counts().head(10).sort_values().plot.barh(color=AZUL)
plt.title("Top 10 categorias por volume de pedidos"); plt.xlabel("Pedidos")
plt.tight_layout(); plt.savefig(f"{FIG}/g05_categorias.png"); plt.close()

plt.figure(figsize=(7,5))  # G6 satisfacao por regiao
(df.groupby("regiao")["satisfeito"].mean()*100).sort_values().plot.bar(color=VERDE)
plt.title("Clientes satisfeitos por região (%)"); plt.ylabel("% satisfeitos")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{FIG}/g06_regiao.png"); plt.close()
print("Graficos G1-G6 salvos.")

# ==========================================================================
# CONCEITO 7 e 9 - CLASSIFICACAO (prever satisfacao) + METRICAS
# ==========================================================================
secao("CONCEITO 7 e 9 - CLASSIFICACAO E METRICAS")
feats = ["tempo_entrega_dias","atraso_dias","frete_total","preco_total","n_itens","parcelas"]
X = pd.DataFrame(StandardScaler().fit_transform(df[feats]), columns=feats)
y = df["satisfeito"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
modelos = {"Regressao Logistica": LogisticRegression(max_iter=1000),
           "Arvore de Decisao": DecisionTreeClassifier(max_depth=5, random_state=42),
           "KNN (k=15)": KNeighborsClassifier(n_neighbors=15)}
res = {}
for nome, m in modelos.items():
    m.fit(Xtr, ytr); pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
    res[nome] = {"modelo": m, "pred": pred, "proba": proba,
                 "Acuracia": accuracy_score(yte, pred), "Precisao": precision_score(yte, pred),
                 "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
                 "AUC": roc_auc_score(yte, proba),
                 "F1_CV": cross_val_score(m, X, y, cv=5, scoring="f1").mean()}
    r = res[nome]
    print(f"\n{nome}\n  Acuracia {r['Acuracia']:.3f} | Precisao {r['Precisao']:.3f} | "
          f"Recall {r['Recall']:.3f} | F1 {r['F1']:.3f} | AUC {r['AUC']:.3f} | F1-CV {r['F1_CV']:.3f}")
melhor_nome = max(res, key=lambda k: res[k]["F1"]); melhor = res[melhor_nome]
print(f"\n>>> Melhor modelo (F1): {melhor_nome}")
print(classification_report(yte, melhor["pred"], target_names=["Insatisfeito", "Satisfeito"]))
ac_tr = accuracy_score(ytr, melhor["modelo"].predict(Xtr))
print(f"Overfitting -> treino {ac_tr:.3f} | teste {melhor['Acuracia']:.3f} | dif {abs(ac_tr-melhor['Acuracia']):.3f}")

plt.figure(figsize=(9,5))  # G7 comparacao modelos
pd.DataFrame({k: {mt: res[k][mt] for mt in ["Acuracia","Precisao","Recall","F1","AUC"]} for k in res}).T.plot.bar(ax=plt.gca())
plt.title("Desempenho dos modelos de classificação"); plt.ylim(0,1); plt.xticks(rotation=10)
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(f"{FIG}/g07_modelos.png"); plt.close()

plt.figure(figsize=(6,5))  # G8 matriz de confusao
sns.heatmap(confusion_matrix(yte, melhor["pred"]), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Insatisf.","Satisf."], yticklabels=["Insatisf.","Satisf."])
plt.title(f"Matriz de confusão — {melhor_nome}"); plt.xlabel("Previsto"); plt.ylabel("Real")
plt.tight_layout(); plt.savefig(f"{FIG}/g08_confusao.png"); plt.close()

plt.figure(figsize=(7,6))  # G9 ROC
for nome in res:
    fpr, tpr, _ = roc_curve(yte, res[nome]["proba"]); plt.plot(fpr, tpr, label=f"{nome} (AUC={res[nome]['AUC']:.2f})")
plt.plot([0,1],[0,1],"k--"); plt.title("Curva ROC dos modelos"); plt.xlabel("Falso Positivo"); plt.ylabel("Verdadeiro Positivo")
plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/g09_roc.png"); plt.close()
print("Graficos G7-G9 salvos.")

# ==========================================================================
# CONCEITO 7 - REGRESSAO LINEAR (prever a nota 1-5)
# ==========================================================================
secao("CONCEITO 7 - REGRESSAO LINEAR (prever a nota)")
yr = df["review_score"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, yr, test_size=0.25, random_state=42)
reg = LinearRegression().fit(Xr_tr, yr_tr); yr_pred = reg.predict(Xr_te)
print(f"  R2 {r2_score(yr_te, yr_pred):.3f} | MAE {mean_absolute_error(yr_te, yr_pred):.2f}")
plt.figure(figsize=(7,5))
imp = pd.Series(np.abs(reg.coef_), index=feats).sort_values()
imp.plot.barh(color="#8172B3")
plt.title("Peso de cada fator na nota (regressão linear)"); plt.tight_layout()
plt.savefig(f"{FIG}/g10_regressao.png"); plt.close()
print("Grafico G10 salvo.")

# ==========================================================================
# CONCEITO 8 - K-MEANS (segmentacao de clientes - RFM)
# ==========================================================================
secao("CONCEITO 8 - CLUSTERIZACAO K-MEANS (segmentacao RFM)")
df["data"] = df["order_purchase_timestamp"]
ref = df["data"].max()
rfm = df.groupby("customer_unique_id").agg(
    Recencia=("data", lambda x: (ref - x.max()).days),
    Frequencia=("order_id", "count"),
    Monetario=("preco_total", "sum")).reset_index()
Xrfm = StandardScaler().fit_transform(rfm[["Recencia","Frequencia","Monetario"]])
km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["cluster"] = km.fit_predict(Xrfm)
print(f"Silhouette: {silhouette_score(Xrfm, rfm.cluster):.3f}")
print(rfm.groupby("cluster")[["Recencia","Frequencia","Monetario"]].mean().round(1))
comp = PCA(n_components=2).fit_transform(Xrfm)
plt.figure(figsize=(8,6))
sns.scatterplot(x=comp[:,0], y=comp[:,1], hue=rfm.cluster, palette="Set2", alpha=0.6)
plt.title("Segmentos de clientes (K-Means)"); plt.xlabel("Componente 1"); plt.ylabel("Componente 2")
plt.tight_layout(); plt.savefig(f"{FIG}/g11_clusters.png"); plt.close()
print("Grafico G11 salvo.")

# ==========================================================================
# CONCEITO 8 - REGRAS DE ASSOCIACAO (APRIORI)
# ==========================================================================
secao("CONCEITO 8 - REGRAS DE ASSOCIACAO (fatores que levam a insatisfacao)")
itens_assoc = pd.DataFrame({
    "Atrasou": df.atrasou == 1,
    "Entrega_lenta": df.tempo_entrega_dias > df.tempo_entrega_dias.median(),
    "Frete_alto": df.frete_total > df.frete_total.median(),
    "Parcelado": df.parcelado == 1,
    "Preco_alto": df.preco_total > df.preco_total.median(),
    "Insatisfeito": df.satisfeito == 0})
freq = apriori(itens_assoc, min_support=0.02, use_colnames=True)
regras = association_rules(freq, metric="confidence", min_threshold=0.35)
regras = regras[regras.consequents.astype(str).str.contains("Insatisfeito")].sort_values("lift", ascending=False)
print("Top regras que levam a 'Insatisfeito':")
print(regras[["antecedents","consequents","support","confidence","lift"]].head(8).to_string(index=False))
if len(regras):
    plt.figure(figsize=(8,6))
    plt.scatter(regras.support, regras.confidence, s=regras.lift*120, alpha=0.6, color=VERM)
    plt.title("Regras de associação (tamanho = lift)"); plt.xlabel("Suporte"); plt.ylabel("Confiança")
    plt.tight_layout(); plt.savefig(f"{FIG}/g12_regras.png"); plt.close()
    print("Grafico G12 salvo.")

# ==========================================================================
secao("RESUMO FINAL")
print(f"Pedidos analisados : {n}")
print(f"KPIs calculados    : {len(kpis)}")
print(f"Graficos gerados   : 12 (pasta figuras/)")
print(f"Melhor modelo      : {melhor_nome} | F1 {melhor['F1']:.3f} | AUC {melhor['AUC']:.3f}")
print(f"Conceitos aplicados: 9 de 9")
print("\nAnalise concluida com sucesso!")
