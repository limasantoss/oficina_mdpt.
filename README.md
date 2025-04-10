# Projeto de Previsão de Temperatura com Redes LSTM

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="40" alt="python logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" height="40" alt="pandas logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="40" alt="numpy logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/matlab/matlab-original.svg" height="40" alt="matlab logo"  />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" height="40" alt="tensorflow logo"  />
</div>

###
## Visão Geral

Este projeto implementa um modelo de previsão de temperatura usando uma rede neural recorrente do tipo Long Short-Term Memory (LSTM). O objetivo é prever a temperatura futura com base em dados históricos simulados. O projeto é desenvolvido em Python, utilizando bibliotecas como NumPy, Pandas, Matplotlib, TensorFlow e Scikit-learn.

## Funcionalidades Principais

* **Geração de Dados Simulados:** Criação de um conjunto de dados de temperatura simulado com padrões sazonais e ruído aleatório.
* **Pré-processamento de Dados:** Normalização dos dados de temperatura utilizando MinMaxScaler para melhorar a performance do modelo.
* **Criação de Janelas Temporais:** Preparação dos dados em sequências temporais para entrada no modelo LSTM.
* **Modelo LSTM:** Implementação de uma rede LSTM multicamadas para modelar a dependência temporal nos dados de temperatura.
* **Treinamento e Validação:** Treinamento do modelo com divisão dos dados em conjuntos de treino e validação, utilizando o otimizador Adam e a função de perda Mean Squared Error.
* **Avaliação do Modelo:** Avaliação do desempenho do modelo nos dados de treino e teste, com visualização das previsões comparadas aos dados reais.
* **Visualização dos Resultados:** Geração de gráficos para visualizar os dados de temperatura, as previsões do modelo e a perda durante o treinamento.

## Tecnologias Utilizadas

* **Python:** Linguagem de programação principal.
* **NumPy:** Para operações numéricas eficientes.
* **Pandas:** Para manipulação e análise de dados.
* **Matplotlib:** Para visualização de dados.
* **TensorFlow:** Framework de aprendizado de máquina para construir e treinar o modelo LSTM.
* **Scikit-learn:** Para pré-processamento de dados (MinMaxScaler).

## Estrutura do Código

O código é organizado da seguinte forma:

1.  **Importação de Bibliotecas:** Importa as bibliotecas necessárias.
2.  **Geração de Dados:** Cria um conjunto de dados simulado de temperatura.
3.  **Pré-processamento:** Normaliza os dados de temperatura.
4.  **Criação de Janelas Temporais:** Prepara os dados para o modelo LSTM.
5.  **Divisão em Treino e Teste:** Divide os dados em conjuntos de treino e teste.
6.  **Construção do Modelo LSTM:** Define a arquitetura do modelo LSTM.
7.  **Treinamento do Modelo:** Treina o modelo com os dados de treino.
8.  **Previsão:** Realiza previsões nos conjuntos de treino e teste.
9.  **Inversão da Normalização:** Desfaz a normalização dos dados previstos.
10. **Visualização dos Resultados:** Plota os dados reais e as previsões.
11. **Visualização da Perda:** Plota a perda durante o treinamento.

## Resultados

O projeto demonstra a capacidade de uma rede LSTM em modelar dados de séries temporais e realizar previsões. Os gráficos gerados mostram a comparação entre os dados reais de temperatura e as previsões do modelo, tanto no conjunto de treino quanto no conjunto de teste. A visualização da perda durante o treinamento ajuda a avaliar a convergência do modelo.

## Como Executar

1.  Certifique-se de ter o Python instalado (versão 3.6 ou superior).
2.  Instale as bibliotecas necessárias usando o `pip`:

    ```bash
    pip install numpy pandas matplotlib tensorflow scikit-learn
    ```

3.  Execute o script Python:

    ```bash
    python seu_script.py  
    ```



