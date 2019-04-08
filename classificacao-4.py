import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np

uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
# Ler csv com o pandas
dados = pd.read_csv(uri)

# Renomar colunas
a_renomear = {
    'mileage_per_year': 'milhas_por_ano',
    'model_year': 'ano_do_modelo',
    'sold': 'vendido',
    'price': 'preco'
}
dados = dados.rename(columns=a_renomear)

# Trocar dados por 0 ou 1
a_trocar = {
    'no': 0,
    'yes': 1
}

dados.vendido = dados.vendido.map(a_trocar)

# Vamos trabalhar com a idade do modelo e não com o ano. Para isso vamos ter uma coluna idade_do_modelo
ano_atual = datetime.today().year
dados['idade_do_modelo'] = ano_atual - dados.ano_do_modelo

# Os dados estão em milhar por anos, vamos converter em km por ano
dados['km_por_ano'] = dados.milhas_por_ano * 1.60934

# Remover colunas que não vamos utilizar, e o normalmente o drop remove linha e queremos remover colunas e para isso o axis=1
dados = dados.drop(columns=['milhas_por_ano', 'ano_do_modelo', 'Unnamed: 0'], axis=1)

# print(dados.head())

x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print("Treinamentos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acuracia for %.2f%%" % acuracia)