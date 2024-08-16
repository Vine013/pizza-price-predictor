import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Carregamento e treinamento do modelo (com tratamento de erro)
try:
    df = pd.read_csv("pizzas.csv")  

    modelo = LinearRegression()
    x = df[["diametro"]]
    y = df[["preco"]]
    modelo.fit(x, y)

    # Interface Web
    st.set_page_config(page_title="Previsor de Preços de Pizza", page_icon="🍕")

    st.title("🍕 Previsor de Preços de Pizza 🍕")
    st.write("Descubra o valor da sua pizza de acordo com o tamanho!")

    # Slider para o diâmetro
    diametro = st.slider("Selecione o diâmetro da pizza (cm):", min_value=5, max_value=50, value=25, step=1)

    # Botão de previsão (com correção no acesso à previsão)
    if st.button("Prever Preço"):
        preco_previsto = modelo.predict([[diametro]]).flatten()[0]
        st.markdown(f"## 💰 O valor da pizza é de R${preco_previsto:.2f} 💰")
        st.balloons()

        # Gráfico de Preços
        diametros = pd.DataFrame({"diametro": range(5, 51)})
        precos = modelo.predict(diametros)
        fig, ax = plt.subplots()
        ax.plot(diametros["diametro"], precos, color="#FF4500")
        
        # Destaque do diâmetro selecionado
        ax.scatter(diametro, preco_previsto, color='yellow', marker='o', s=110, zorder=5)  # Ponto de destaque

        ax.set_xlabel("Diâmetro (cm)")
        ax.set_ylabel("Preço (R$)")
        st.pyplot(fig)

    # Sobre o Modelo 
    st.subheader("Sobre o Modelo")
    st.write("Este modelo utiliza regressão linear para prever o preço da pizza com base no diâmetro. Foi treinado com dados de pizzarias locais.")

    st.markdown("---")
    st.write("**Vinicius Santiago**")
    st.write("[LinkedIn](https://www.linkedin.com/in/vinicius-santiago-aab4851ab/) | [GitHub](https://github.com/Vine013)")

except FileNotFoundError:
    st.error("Arquivo 'pizzas.csv' não encontrado. Verifique se o arquivo existe no mesmo diretório do script.")
except Exception as e:
    st.error(f"Ocorreu um erro: {e}")