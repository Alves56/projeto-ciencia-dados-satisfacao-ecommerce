# -*- coding: utf-8 -*-
"""
Baixa as tabelas do conjunto publico Olist (e-commerce brasileiro) para data/.
Rode uma unica vez:  python scripts/preparar_dados.py
Fonte: Brazilian E-Commerce Public Dataset by Olist (Kaggle) - espelho GitHub.
"""
import os
import urllib.request

# Roda sempre a partir da raiz do projeto (mesmo sendo chamado de dentro de scripts/)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASTA = "data"
os.makedirs(PASTA, exist_ok=True)

BASE = "https://raw.githubusercontent.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist/master/files/"
ARQUIVOS = {
    "olist_orders_dataset.csv": "pedidos",
    "olist_order_reviews_dataset.csv": "avaliacoes",
    "olist_order_items_dataset.csv": "itens",
    "olist_order_payments_dataset.csv": "pagamentos",
    "olist_products_dataset.csv": "produtos",
    "olist_customers_dataset.csv": "clientes",
    "product_category_name_translation.csv": "traducao de categorias",
}

for arquivo, descricao in ARQUIVOS.items():
    destino = os.path.join(PASTA, arquivo)
    print(f"Baixando {descricao} ...")
    urllib.request.urlretrieve(BASE + arquivo, destino)
    tamanho = os.path.getsize(destino) // 1024
    print(f"  OK -> {destino}  ({tamanho} KB)")

print("\nPronto! Todas as tabelas do Olist estao na pasta data/.")
