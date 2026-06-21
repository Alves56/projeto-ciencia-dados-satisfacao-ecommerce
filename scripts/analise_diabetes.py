# -*- coding: utf-8 -*-
"""
==============================================================================
PROJETO PRATICO DE CIENCIA DE DADOS  -  TEMA: DIABETES
==============================================================================
Disciplina: Ciencia de Dados | CEUB | Turma B
Datasets publicos:
  1) Pima Indians Diabetes (classificacao)      -> data/pima_diabetes.csv
  2) Diabetes Progression / Efron et al. (regr) -> data/diabetes_progressao.csv

Conceitos de Ciencia de Dados aplicados (9 de 9):
  1. Coleta de Dados
  2. Limpeza, Pre-processamento e Integracao
  3. Estatistica Descritiva
  4. Construcao de Indicadores (KPIs)  -> 20+
  5. Visualizacao de Dados / Storytelling -> 12+ graficos
  6. Feature Engineering
  7. Modelagem Preditiva (Reg. Logistica, Arvore, KNN, Reg. Linear)
  8. Machine Learning (K-Means + Regras de Associacao Apriori)
  9. Metricas de Avaliacao (Precision/Recall/F1, Matriz de Confusao, CV, ROC)
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # salva figuras sem precisar de tela
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    r2_score, mean_absolute_error, silhouette_score,
)
from mlxtend.frequent_patterns import apriori, association_rules

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

# Roda sempre a partir da raiz do projeto (mesmo sendo chamado de dentro de scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIG = "figuras"
os.makedirs(FIG, exist_ok=True)  # cria a pasta figuras/ se ela ainda nao existir

# Garante que os datasets existem; se nao, orienta a rodar o preparador
if not (os.path.exists("data/pima_diabetes.csv") and os.path.exists("data/diabetes_progressao.csv")):
    raise SystemExit("Datasets nao encontrados. Rode antes:  python preparar_dados.py")

def secao(titulo):
    print("\n" + "=" * 78)
    print(titulo)
    print("=" * 78)


# ==========================================================================
# CONCEITO 1 - COLETA DE DADOS
# ==========================================================================
secao("CONCEITO 1 - COLETA DE DADOS")

pima = pd.read_csv("data/pima_diabetes.csv")
prog = pd.read_csv("data/diabetes_progressao.csv")

print(f"Dataset 1 (Pima)        : {pima.shape[0]} linhas x {pima.shape[1]} colunas")
print(f"Dataset 2 (Progressao)  : {prog.shape[0]} linhas x {prog.shape[1]} colunas")
print("\nPrimeiras linhas do Dataset 1:")
print(pima.head())


# ==========================================================================
# CONCEITO 2 - LIMPEZA, PRE-PROCESSAMENTO E INTEGRACAO
# ==========================================================================
secao("CONCEITO 2 - LIMPEZA, PRE-PROCESSAMENTO E INTEGRACAO")

# No Pima, zeros em variaveis biologicas sao impossiveis -> sao valores ausentes
cols_zero_invalido = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("Valores 0 (ausentes disfarcados) antes da limpeza:")
print((pima[cols_zero_invalido] == 0).sum())

pima[cols_zero_invalido] = pima[cols_zero_invalido].replace(0, np.nan)

# Imputacao pela mediana (robusta a outliers)
for c in cols_zero_invalido:
    pima[c] = pima[c].fillna(pima[c].median())

print("\nValores ausentes apos imputacao (deve ser 0):")
print(pima.isna().sum().sum())

# --- INTEGRACAO DOS DOIS DATASETS ---------------------------------------
# Colunas em comum (mesmo conceito clinico) entre os dois datasets:
#   Pima: Age, BMI, BloodPressure, Glucose
#   Prog: Idade, IMC, PressaoArterial, Glicemia
coorte_pima = pima[["Age", "BMI", "BloodPressure", "Glucose"]].copy()
coorte_pima.columns = ["Idade", "IMC", "PressaoArterial", "Glicemia"]
coorte_pima["Fonte"] = "Pima"

coorte_prog = prog[["Idade", "IMC", "PressaoArterial", "Glicemia"]].copy()
coorte_prog["Fonte"] = "Progressao"

coorte = pd.concat([coorte_pima, coorte_prog], ignore_index=True)
print(f"\nCoorte integrada (uniao dos 2 datasets): {coorte.shape[0]} pacientes")
print(coorte.groupby("Fonte").size())

# Normalizacao (StandardScaler) - usada nos modelos sensiveis a escala
escala = StandardScaler()
X_pima_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
pima_norm = pd.DataFrame(escala.fit_transform(pima[X_pima_cols]), columns=X_pima_cols)
print("\nExemplo de dados normalizados (media ~0, desvio ~1):")
print(pima_norm.describe().loc[["mean", "std"]].round(2))


# ==========================================================================
# CONCEITO 6 - FEATURE ENGINEERING  (feito cedo p/ alimentar KPIs e graficos)
# ==========================================================================
secao("CONCEITO 6 - FEATURE ENGINEERING")

pima["FaixaEtaria"] = pd.cut(pima["Age"], bins=[20, 30, 40, 50, 100],
                             labels=["21-30", "31-40", "41-50", "50+"],
                             right=True, include_lowest=True)

pima["CategoriaIMC"] = pd.cut(pima["BMI"], bins=[0, 18.5, 25, 30, 100],
                              labels=["Abaixo", "Normal", "Sobrepeso", "Obeso"])

pima["NivelGlicose"] = pd.cut(pima["Glucose"], bins=[0, 99, 125, 300],
                              labels=["Normal", "Pre-diabetes", "Diabetes"])

pima["MuitasGestacoes"] = (pima["Pregnancies"] >= 5).astype(int)
# Variavel de interacao: carga metabolica (glicose * imc normalizados)
pima["CargaMetabolica"] = (pima["Glucose"] * pima["BMI"]) / 1000

print("Novas variaveis criadas: FaixaEtaria, CategoriaIMC, NivelGlicose, "
      "MuitasGestacoes, CargaMetabolica")
print(pima[["Age", "FaixaEtaria", "BMI", "CategoriaIMC",
            "Glucose", "NivelGlicose"]].head())


# ==========================================================================
# CONCEITO 3 - ESTATISTICA DESCRITIVA
# ==========================================================================
secao("CONCEITO 3 - ESTATISTICA DESCRITIVA")

print("Resumo estatistico (Dataset 1 - Pima):")
print(pima[X_pima_cols].describe().round(2))

print("\nMedidas de tendencia central e dispersao - Glicose:")
print(f"  Media   : {pima['Glucose'].mean():.2f}")
print(f"  Mediana : {pima['Glucose'].median():.2f}")
print(f"  Moda    : {pima['Glucose'].mode()[0]:.2f}")
print(f"  Desvio  : {pima['Glucose'].std():.2f}")


# ==========================================================================
# CONCEITO 4 - INDICADORES (KPIs)  -> 25 KPIs
# ==========================================================================
secao("CONCEITO 4 - INDICADORES (KPIs)")

n = len(pima)
diab = pima[pima["Outcome"] == 1]
ndiab = pima[pima["Outcome"] == 0]
obesos = pima[pima["BMI"] >= 30]
nao_obesos = pima[pima["BMI"] < 30]

kpis = {
    "01. Total de pacientes": n,
    "02. Pacientes diabeticos": len(diab),
    "03. Pacientes nao-diabeticos": len(ndiab),
    "04. Prevalencia de diabetes (%)": 100 * len(diab) / n,
    "05. Idade media (anos)": pima["Age"].mean(),
    "06. Idade mediana (anos)": pima["Age"].median(),
    "07. Glicose media geral": pima["Glucose"].mean(),
    "08. Glicose media - diabeticos": diab["Glucose"].mean(),
    "09. Glicose media - nao diabeticos": ndiab["Glucose"].mean(),
    "10. IMC medio": pima["BMI"].mean(),
    "11. Pacientes obesos (%)": 100 * len(obesos) / n,
    "12. Glicose alta >=126 (%)": 100 * (pima["Glucose"] >= 126).mean(),
    "13. Pressao arterial media": pima["BloodPressure"].mean(),
    "14. Insulina media": pima["Insulin"].mean(),
    "15. Media de gestacoes": pima["Pregnancies"].mean(),
    "16. Hist. familiar medio (DPF)": pima["DiabetesPedigreeFunction"].mean(),
    "17. Desvio padrao da glicose": pima["Glucose"].std(),
    "18. Faixa etaria mais comum": pima["FaixaEtaria"].mode()[0],
    "19. Correlacao Glicose-Diabetes": pima["Glucose"].corr(pima["Outcome"]),
    "20. Taxa diabetes em obesos (%)": 100 * obesos["Outcome"].mean(),
    "21. Taxa diabetes em nao-obesos (%)": 100 * nao_obesos["Outcome"].mean(),
    "22. Pacientes com >=5 gestacoes (%)": 100 * pima["MuitasGestacoes"].mean(),
    "23. Progressao media da doenca (DS2)": prog["ProgressaoDoenca"].mean(),
    "24. Correlacao IMC-Progressao (DS2)": prog["IMC"].corr(prog["ProgressaoDoenca"]),
    "25. Glicemia media coorte integrada": coorte["Glicemia"].mean(),
}

for nome, valor in kpis.items():
    if isinstance(valor, float):
        print(f"  {nome:42s}: {valor:.2f}")
    else:
        print(f"  {nome:42s}: {valor}")


# ==========================================================================
# CONCEITO 5 - VISUALIZACAO DE DADOS / STORYTELLING  -> 12 graficos
# ==========================================================================
secao("CONCEITO 5 - VISUALIZACAO DE DADOS (gerando figuras em figuras/)")

# G1 - Distribuicao da prevalencia (pizza)
plt.figure(figsize=(6, 6))
pima["Outcome"].map({0: "Nao-diabetico", 1: "Diabetico"}).value_counts().plot.pie(
    autopct="%1.1f%%", startangle=90, colors=["#4C72B0", "#C44E52"])
plt.title("G1 - Prevalencia de Diabetes na amostra")
plt.ylabel("")
plt.tight_layout(); plt.savefig(f"{FIG}/g01_prevalencia.png"); plt.close()

# G2 - Histograma da glicose por grupo
plt.figure(figsize=(8, 5))
sns.histplot(data=pima, x="Glucose", hue="Outcome", kde=True, bins=30,
             palette={0: "#4C72B0", 1: "#C44E52"})
plt.title("G2 - Distribuicao da Glicose por diagnostico")
plt.xlabel("Glicose"); plt.tight_layout()
plt.savefig(f"{FIG}/g02_hist_glicose.png"); plt.close()

# G3 - Boxplot glicose x outcome
plt.figure(figsize=(7, 5))
sns.boxplot(data=pima, x="Outcome", y="Glucose", palette=["#4C72B0", "#C44E52"])
plt.xticks([0, 1], ["Nao-diabetico", "Diabetico"])
plt.title("G3 - Boxplot da Glicose por diagnostico"); plt.tight_layout()
plt.savefig(f"{FIG}/g03_box_glicose.png"); plt.close()

# G4 - Heatmap de correlacao
plt.figure(figsize=(9, 7))
sns.heatmap(pima[X_pima_cols + ["Outcome"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0)
plt.title("G4 - Matriz de Correlacao"); plt.tight_layout()
plt.savefig(f"{FIG}/g04_correlacao.png"); plt.close()

# G5 - Taxa de diabetes por faixa etaria
plt.figure(figsize=(8, 5))
taxa_idade = pima.groupby("FaixaEtaria")["Outcome"].mean() * 100
taxa_idade.plot.bar(color="#C44E52")
plt.title("G5 - Taxa de Diabetes por Faixa Etaria")
plt.ylabel("% diabeticos"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f"{FIG}/g05_taxa_idade.png"); plt.close()

# G6 - Taxa de diabetes por categoria de IMC
plt.figure(figsize=(8, 5))
taxa_imc = pima.groupby("CategoriaIMC")["Outcome"].mean() * 100
taxa_imc.plot.bar(color="#55A868")
plt.title("G6 - Taxa de Diabetes por Categoria de IMC")
plt.ylabel("% diabeticos"); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f"{FIG}/g06_taxa_imc.png"); plt.close()

# G7 - Scatter glicose x IMC
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pima, x="Glucose", y="BMI", hue="Outcome",
                palette={0: "#4C72B0", 1: "#C44E52"}, alpha=0.7)
plt.title("G7 - Glicose vs IMC por diagnostico"); plt.tight_layout()
plt.savefig(f"{FIG}/g07_scatter.png"); plt.close()

# G8 - Pairplot (subconjunto)
g = sns.pairplot(pima[["Glucose", "BMI", "Age", "Outcome"]], hue="Outcome",
                 palette={0: "#4C72B0", 1: "#C44E52"})
g.fig.suptitle("G8 - Relacoes entre variaveis-chave", y=1.02)
g.savefig(f"{FIG}/g08_pairplot.png"); plt.close()

print("Graficos G1-G8 (exploratorios) salvos. (G9-G12 saem nos modelos.)")


# ==========================================================================
# CONCEITO 7 - MODELAGEM PREDITIVA  +  CONCEITO 9 - METRICAS
# ==========================================================================
secao("CONCEITO 7 e 9 - CLASSIFICACAO (prever diabetes) E METRICAS")

X = pima_norm  # features normalizadas
y = pima["Outcome"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)

modelos = {
    "Regressao Logistica": LogisticRegression(max_iter=1000),
    "Arvore de Decisao": DecisionTreeClassifier(max_depth=4, random_state=42),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11),
}

resultados = {}
for nome, modelo in modelos.items():
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    proba = modelo.predict_proba(X_te)[:, 1]
    cv = cross_val_score(modelo, X, y, cv=5, scoring="f1").mean()  # validacao cruzada
    resultados[nome] = {
        "modelo": modelo,
        "Acuracia": accuracy_score(y_te, pred),
        "Precisao": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba),
        "F1_CV": cv,
        "pred": pred, "proba": proba,
    }
    print(f"\n{nome}")
    print(f"  Acuracia : {resultados[nome]['Acuracia']:.3f}")
    print(f"  Precisao : {resultados[nome]['Precisao']:.3f}")
    print(f"  Recall   : {resultados[nome]['Recall']:.3f}")
    print(f"  F1-Score : {resultados[nome]['F1']:.3f}")
    print(f"  AUC-ROC  : {resultados[nome]['AUC']:.3f}")
    print(f"  F1 (validacao cruzada 5-fold): {cv:.3f}")

melhor_nome = max(resultados, key=lambda k: resultados[k]["F1"])
melhor = resultados[melhor_nome]
print(f"\n>>> Melhor modelo (F1): {melhor_nome}")
print("\nRelatorio de classificacao do melhor modelo:")
print(classification_report(y_te, melhor["pred"],
                            target_names=["Nao-diabetico", "Diabetico"]))

# Verificacao de overfitting (treino vs teste) no melhor modelo
ac_tr = accuracy_score(y_tr, melhor["modelo"].predict(X_tr))
ac_te = melhor["Acuracia"]
print(f"Overfitting check -> Acuracia treino: {ac_tr:.3f} | teste: {ac_te:.3f} "
      f"| diferenca: {abs(ac_tr - ac_te):.3f}")

# G9 - Comparacao de modelos
plt.figure(figsize=(9, 5))
metr = pd.DataFrame({k: {m: resultados[k][m] for m in
                     ["Acuracia", "Precisao", "Recall", "F1", "AUC"]}
                     for k in resultados}).T
metr.plot.bar(ax=plt.gca())
plt.title("G9 - Comparacao de desempenho dos modelos")
plt.ylabel("Pontuacao"); plt.xticks(rotation=15); plt.ylim(0, 1)
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(f"{FIG}/g09_comparacao_modelos.png"); plt.close()

# G10 - Matriz de confusao do melhor modelo
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_te, melhor["pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Nao-diab", "Diab"], yticklabels=["Nao-diab", "Diab"])
plt.title(f"G10 - Matriz de Confusao ({melhor_nome})")
plt.xlabel("Previsto"); plt.ylabel("Real"); plt.tight_layout()
plt.savefig(f"{FIG}/g10_matriz_confusao.png"); plt.close()

# G11 - Curvas ROC
plt.figure(figsize=(7, 6))
for nome in resultados:
    fpr, tpr, _ = roc_curve(y_te, resultados[nome]["proba"])
    plt.plot(fpr, tpr, label=f"{nome} (AUC={resultados[nome]['AUC']:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("G11 - Curva ROC"); plt.xlabel("Falso Positivo"); plt.ylabel("Verdadeiro Positivo")
plt.legend(); plt.tight_layout()
plt.savefig(f"{FIG}/g11_roc.png"); plt.close()

# Importancia de variaveis (arvore)
arvore = resultados["Arvore de Decisao"]["modelo"]
imp = pd.Series(arvore.feature_importances_, index=X_pima_cols).sort_values()
plt.figure(figsize=(8, 5))
imp.plot.barh(color="#8172B3")
plt.title("G12 - Importancia das variaveis (Arvore de Decisao)")
plt.tight_layout(); plt.savefig(f"{FIG}/g12_importancia.png"); plt.close()
print("\nGraficos G9-G12 (modelos) salvos.")


# ==========================================================================
# CONCEITO 7 (cont.) - REGRESSAO LINEAR (Dataset 2)
# ==========================================================================
secao("CONCEITO 7 - REGRESSAO LINEAR (prever progressao da doenca - DS2)")

feat_reg = ["Idade", "IMC", "PressaoArterial", "ColesterolTotal",
            "LDL", "HDL", "Triglicerides", "Glicemia"]
Xr = prog[feat_reg]
yr = prog["ProgressaoDoenca"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=42)

reg = LinearRegression().fit(Xr_tr, yr_tr)
yr_pred = reg.predict(Xr_te)
print(f"  R2  : {r2_score(yr_te, yr_pred):.3f}")
print(f"  MAE : {mean_absolute_error(yr_te, yr_pred):.2f}")

# G13 - Real vs Previsto
plt.figure(figsize=(7, 6))
plt.scatter(yr_te, yr_pred, alpha=0.6, color="#4C72B0")
plt.plot([yr_te.min(), yr_te.max()], [yr_te.min(), yr_te.max()], "r--")
plt.title("G13 - Regressao Linear: Real vs Previsto")
plt.xlabel("Progressao real"); plt.ylabel("Progressao prevista")
plt.tight_layout(); plt.savefig(f"{FIG}/g13_regressao.png"); plt.close()
print("Grafico G13 salvo.")


# ==========================================================================
# CONCEITO 8 - MACHINE LEARNING: K-MEANS (clusterizacao)
# ==========================================================================
secao("CONCEITO 8 - CLUSTERIZACAO K-MEANS")

X_clu = pima_norm.copy()
km = KMeans(n_clusters=3, random_state=42, n_init=10)
pima["Cluster"] = km.fit_predict(X_clu)
sil = silhouette_score(X_clu, pima["Cluster"])
print(f"  Numero de clusters: 3 | Silhouette score: {sil:.3f}")
print("\n  Perfil medio de cada cluster:")
print(pima.groupby("Cluster")[["Glucose", "BMI", "Age", "Outcome"]].mean().round(2))

# G14 - Clusters em 2D (PCA)
pca = PCA(n_components=2)
comp = pca.fit_transform(X_clu)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=comp[:, 0], y=comp[:, 1], hue=pima["Cluster"],
                palette="Set2", alpha=0.8)
plt.title("G14 - Clusters de pacientes (K-Means + PCA)")
plt.xlabel("Componente 1"); plt.ylabel("Componente 2")
plt.tight_layout(); plt.savefig(f"{FIG}/g14_clusters.png"); plt.close()
print("Grafico G14 salvo.")


# ==========================================================================
# CONCEITO 8 - REGRAS DE ASSOCIACAO (APRIORI)
# ==========================================================================
secao("CONCEITO 8 - REGRAS DE ASSOCIACAO (APRIORI)")

# Transforma variaveis em itens binarios (one-hot)
itens = pd.DataFrame()
itens["Glicose_alta"] = pima["Glucose"] >= 126
itens["Obeso"] = pima["BMI"] >= 30
itens["Idade_40+"] = pima["Age"] >= 40
itens["Muitas_gestacoes"] = pima["Pregnancies"] >= 5
itens["Hist_familiar_alto"] = pima["DiabetesPedigreeFunction"] >= pima["DiabetesPedigreeFunction"].median()
itens["Diabetico"] = pima["Outcome"] == 1

freq = apriori(itens, min_support=0.10, use_colnames=True)
regras = association_rules(freq, metric="confidence", min_threshold=0.5)
regras = regras[regras["consequents"].astype(str).str.contains("Diabetico")]
regras = regras.sort_values("lift", ascending=False)

print("Top regras que levam a 'Diabetico':")
cols_show = ["antecedents", "consequents", "support", "confidence", "lift"]
print(regras[cols_show].head(8).to_string(index=False))

# G15 - Suporte vs Confianca das regras
if len(regras) > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(regras["support"], regras["confidence"],
                s=regras["lift"] * 80, alpha=0.6, color="#C44E52")
    plt.title("G15 - Regras de Associacao (tamanho = lift)")
    plt.xlabel("Suporte"); plt.ylabel("Confianca")
    plt.tight_layout(); plt.savefig(f"{FIG}/g15_regras.png"); plt.close()
    print("Grafico G15 salvo.")


# ==========================================================================
# ENCERRAMENTO
# ==========================================================================
secao("RESUMO FINAL")
print(f"Datasets analisados      : 2 (Pima + Progressao)")
print(f"KPIs calculados          : {len(kpis)}")
print(f"Graficos gerados         : 15 (pasta figuras/)")
print(f"Melhor modelo (classif.) : {melhor_nome} | F1 = {melhor['F1']:.3f} | AUC = {melhor['AUC']:.3f}")
print(f"Conceitos de CD aplicados: 9 de 9")
print("\nAnalise concluida com sucesso!")
