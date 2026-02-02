import streamlit as st
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIGURAÇÃO INICIAL
# ======================
st.set_page_config(page_title="Previsão de Score de Crédito", layout="centered")

st.title("🔮 Previsão de Score de Crédito")

# ======================
# CARREGAR BASE DE DADOS
# ======================
CAMINHO_BASE = os.path.dirname(__file__)
CAMINHO_CSV = os.path.join(CAMINHO_BASE, "clientes.csv")

try:
    tabela = pd.read_csv(
    CAMINHO_CSV,
    sep=";",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

except Exception as e:
    st.error("Erro ao carregar o arquivo clientes.csv")
    st.exception(e)
    st.stop()

st.subheader("📄 Prévia da base de dados")
st.dataframe(tabela.head())

# ======================
# TRATAMENTO DOS DADOS
# ======================
cod_profissao = LabelEncoder()
cod_mix = LabelEncoder()
cod_pagamento = LabelEncoder()

tabela["profissao"] = cod_profissao.fit_transform(tabela["profissao"])
tabela["mix"] = cod_mix.fit_transform(tabela["mix"])
tabela["comportamento_pagamento"] = cod_pagamento.fit_transform(
    tabela["comportamento_pagamento"]
)

y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"])

# ======================
# TREINAMENTO DO MODELO
# ======================
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=42
)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(x_treino, y_treino)

# ======================
# INTERFACE DO USUÁRIO
# ======================
st.subheader("🧾 Dados do cliente")

profissao = st.selectbox(
    "Profissão",
    cod_profissao.classes_
)

mix = st.selectbox(
    "Mix",
    cod_mix.classes_
)

comportamento = st.selectbox(
    "Comportamento de pagamento",
    cod_pagamento.classes_
)

if st.button("🔍 Prever score"):
    novo_cliente = pd.DataFrame([{
        "profissao": cod_profissao.transform([profissao])[0],
        "mix": cod_mix.transform([mix])[0],
        "comportamento_pagamento": cod_pagamento.transform([comportamento])[0]
    }])

    previsao = modelo.predict(novo_cliente)[0]

    st.success(f"✅ Score de crédito previsto: **{previsao}**")
