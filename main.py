import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================
# CARREGAR E TREINAR MODELO
# ======================
tabela = pd.read_csv("clientes.csv")

# Codificadores
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

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=0.3
)

modelo = RandomForestClassifier()
modelo.fit(x_treino, y_treino)

# ======================
# INTERFACE DO USUÁRIO
# ======================
st.title("🔮 Previsão de Score de Crédito")

st.write("Preencha os dados do cliente:")

profissao = st.selectbox(
    "Profissão",
    cod_profissao.classes_
)

mix = st.selectbox(
    "Mix",
    cod_mix.classes_
)

comportamento = st.selectbox(
    "Comportamento de Pagamento",
    cod_pagamento.classes_
)

if st.button("Prever score"):
    novo_cliente = pd.DataFrame([{
        "profissao": cod_profissao.transform([profissao])[0],
        "mix": cod_mix.transform([mix])[0],
        "comportamento_pagamento": cod_pagamento.transform([comportamento])[0]
    }])

    previsao = modelo.predict(novo_cliente)

    st.success(f"Score de crédito previsto: **{previsao[0]}**")
