import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# Reprodutibilidade
np.random.seed(42)

# Geração de dados (simulados)
tempo = np.linspace(0, 30, 200)
temperatura = (
    np.sin(tempo) * 15
    + np.cos(tempo * 3) * 5
    + 25
    + np.random.normal(0, 2, 200)
)
dados = pd.DataFrame(temperatura, columns=["Temperatura"])

# Pré-processamento: Normalização
escalador = MinMaxScaler(feature_range=(0, 1))
temperatura_escalada = escalador.fit_transform(dados[["Temperatura"]])

# Função para criar janelas temporais
def criar_janelas(dados, tamanho_janela):
    X, y = [], []
    for i in range(len(dados) - tamanho_janela):
        janela = dados[i : (i + tamanho_janela), 0]
        X.append(janela)
        y.append(dados[i + tamanho_janela, 0])
    return np.array(X), np.array(y)

tamanho_janela = 20
X, y = criar_janelas(temperatura_escalada, tamanho_janela)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Divisão treino/teste
tamanho_treino = int(len(X) * 0.8)
X_treino, X_teste = X[0:tamanho_treino], X[tamanho_treino:len(X)]
y_treino, y_teste = y[0:tamanho_treino], y[tamanho_treino:len(y)]


# Construção do modelo LSTM
modelo = keras.Sequential()
modelo.add(
    keras.layers.LSTM(
        50, return_sequences=True, input_shape=(X_treino.shape[1], 1)
    )
)
modelo.add(keras.layers.Dropout(0.2))
modelo.add(keras.layers.LSTM(50, return_sequences=True))
modelo.add(keras.layers.Dropout(0.2))
modelo.add(keras.layers.LSTM(50))
modelo.add(keras.layers.Dropout(0.2))
modelo.add(keras.layers.Dense(1))
modelo.compile(optimizer="adam", loss="mean_squared_error")

# Treinamento
historico = modelo.fit(
    X_treino,
    y_treino,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
	shuffle = False
)

# Previsões
previsao_treino = modelo.predict(X_treino)
previsao_teste = modelo.predict(X_teste)

# Inversão da normalização
previsao_treino = escalador.inverse_transform(previsao_treino)
y_treino_original = escalador.inverse_transform(y_treino.reshape(-1, 1))
previsao_teste = escalador.inverse_transform(previsao_teste)
y_teste_original = escalador.inverse_transform(y_teste.reshape(-1, 1))

# Visualização
plt.figure(figsize=(12, 6))
plt.plot(dados["Temperatura"].values, label="Temperatura Real", color="blue")

previsao_treino_plot = np.empty_like(dados["Temperatura"])
previsao_treino_plot[:] = np.nan
previsao_treino_plot[
    tamanho_janela : len(previsao_treino) + tamanho_janela
] = previsao_treino.flatten()
plt.plot(previsao_treino_plot, label="Previsão no Treino", color="green")

previsao_teste_plot = np.empty_like(dados["Temperatura"])
previsao_teste_plot[:] = np.nan
previsao_teste_plot[
    len(previsao_treino) + tamanho_janela : len(dados)
] = previsao_teste.flatten()
plt.plot(previsao_teste_plot, label="Previsão no Teste", color="red")

plt.legend()
plt.title("Previsão da Temperatura (Rede Neural)")
plt.xlabel("Dias")
plt.ylabel("Temperatura (°C)")
plt.grid(True)
plt.show()

# Visualização da perda
plt.figure(figsize=(8, 4))
plt.plot(historico.history['loss'], label='Perda no Treino')
plt.plot(historico.history['val_loss'], label='Perda na Validação')
plt.title('Perda durante o Treino')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.show()