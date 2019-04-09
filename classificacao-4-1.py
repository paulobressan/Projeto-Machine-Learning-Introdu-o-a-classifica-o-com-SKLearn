import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
# O standardScale vai preprocessar os dados separando o treino_x e o teste_x de acordo com a disperção da media e dividir pelo desvio padrão
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Estimador de acuracia. Vamos usar para termos uma base de dados onde podemos estimar a acuracia do modelo
from sklearn.dummy import DummyClassifier
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

# Semente para manter uma sequencia de numeros randomicos iguais
SEED = 5
np.random.seed(SEED)
# Extraindo o treino e 25% dos dados em testes
# O stratify, se a variável y for uma variável categórica binária com valores 0 e 1 e houver 25% de zeros e 75% de uns, stratify = y garantirá que sua divisão aleatória tenha 25% de 0s e 75% de 1s.
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

# O Scaler vai pegar os dados e analisar a media dos dados e suas disperção e dividir pelo desvio padrão
scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

print("Treinamentos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acuracia for %.2f%%" % acuracia)
