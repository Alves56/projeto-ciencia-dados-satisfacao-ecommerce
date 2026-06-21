"""
Baixa/gera os dois datasets publicos usados no projeto e salva em data/.
Rode uma unica vez:  python preparar_dados.py
"""
import os
import urllib.request
import pandas as pd
from sklearn.datasets import load_diabetes

# Roda sempre a partir da raiz do projeto (mesmo sendo chamado de dentro de scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASTA = "data"
os.makedirs(PASTA, exist_ok=True)  # cria a pasta data/ se ela ainda nao existir

# ---------------------------------------------------------------------------
# DATASET 1 - Pima Indians Diabetes Database (classificacao)
# Fonte original: National Institute of Diabetes and Digestive and Kidney Diseases
# Mirror estavel: github.com/jbrownlee/Datasets
# ---------------------------------------------------------------------------
URL_PIMA = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUNAS_PIMA = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

print("Baixando Pima Indians Diabetes...")
urllib.request.urlretrieve(URL_PIMA, f"{PASTA}/pima_diabetes.csv")
pima = pd.read_csv(f"{PASTA}/pima_diabetes.csv", header=None, names=COLUNAS_PIMA)
pima.to_csv(f"{PASTA}/pima_diabetes.csv", index=False)
print(f"  OK -> {PASTA}/pima_diabetes.csv  ({pima.shape[0]} linhas, {pima.shape[1]} colunas)")

# ---------------------------------------------------------------------------
# DATASET 2 - Diabetes Progression (Efron et al., 2004) via scikit-learn
# 442 pacientes; alvo = progressao da doenca apos 1 ano (regressao)
# scaled=False -> valores em unidades reais (idade em anos, bmi real, etc.)
# ---------------------------------------------------------------------------
print("Gerando Diabetes Progression (scikit-learn)...")
diab = load_diabetes(as_frame=True, scaled=False)
df2 = diab.frame.copy()
df2 = df2.rename(columns={
    "age": "Idade", "sex": "Sexo", "bmi": "IMC", "bp": "PressaoArterial",
    "s1": "ColesterolTotal", "s2": "LDL", "s3": "HDL", "s4": "TCH",
    "s5": "Triglicerides", "s6": "Glicemia", "target": "ProgressaoDoenca",
})
df2.to_csv(f"{PASTA}/diabetes_progressao.csv", index=False)
print(f"  OK -> {PASTA}/diabetes_progressao.csv  ({df2.shape[0]} linhas, {df2.shape[1]} colunas)")

print("\nPronto! Os dois datasets estao na pasta data/.")
