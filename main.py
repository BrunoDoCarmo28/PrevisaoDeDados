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
# CARREGAR BASE
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

# ======================
# PADRONIZAR TEXTO
# ======================
tabela["profissao"] = tabela["profissao"].str.strip().str.title()
tabela["mix"] = tabela["mix"].str.strip().str.title()
tabela["comportamento_pagamento"] = tabela["comportamento_pagamento"].str.strip().str.title()

# ======================
# MOSTRAR APENAS 5 LINHAS
# ======================
st.subheader("📄 Prévia da base")
st.dataframe(tabela.head())

# ======================
# TREINAMENTO
# ======================
cod_profissao = LabelEncoder()
cod_mix = LabelEncoder()
cod_pagamento = LabelEncoder()

tabela["profissao_cod"] = cod_profissao.fit_transform(tabela["profissao"])
tabela["mix_cod"] = cod_mix.fit_transform(tabela["mix"])
tabela["pagamento_cod"] = cod_pagamento.fit_transform(
    tabela["comportamento_pagamento"]
)

x = tabela[["profissao_cod", "mix_cod", "pagamento_cod"]]
y = tabela["score_credito"]

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3, random_state=42
)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(x_treino, y_treino)

# ======================
# INTERFACE
# ======================
st.subheader("🧾 Dados do cliente")

# 🔥 AQUI ESTÁ A DIFERENÇA:
# Agora o select pega TODAS as profissões da tabela inteira

profissao_escolhida = st.selectbox(
    "Profissão",
    sorted(tabela["profissao"].unique())
)

# Buscar dados da profissão escolhida NA BASE INTEIRA
dados_cliente = tabela[
    tabela["profissao"] == profissao_escolhida
].iloc[0]

mix_auto = dados_cliente["mix"]
pagamento_auto = dados_cliente["comportamento_pagamento"]

st.write("📌 Mix automático:", mix_auto)
st.write("📌 Comportamento automático:", pagamento_auto)

# ======================
# PREVISÃO
# ======================
if st.button("🔍 Prever score"):

    novo_cliente = pd.DataFrame([{
        "profissao_cod": cod_profissao.transform([profissao_escolhida])[0],
        "mix_cod": cod_mix.transform([mix_auto])[0],
        "pagamento_cod": cod_pagamento.transform([pagamento_auto])[0]
    }])

    previsao = modelo.predict(novo_cliente)[0]

    st.success(f"✅ Score de crédito previsto: **{previsao}**")
