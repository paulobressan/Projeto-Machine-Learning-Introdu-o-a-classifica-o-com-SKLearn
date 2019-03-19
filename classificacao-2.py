'''
PROJETO 2, VAMOS ANALISAR OS DADOS GERADO ATRAVES DA NAVEGAÇÃO DE UM CLIENTE EM UM SITE
'''

# Bilbioteca de analise de dados
import pandas as pd
# sklearn para treinar modelos
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
# Split dos dados x e y para separar treino de teste
from sklearn.model_selection import train_test_split

# O train_test_split separa dos dados com uma semente randomica. Para isso vamos definir uma semente para que
# o resultado seja sempre o mesmo, ou seja ele sempre vai separar da mesma forma os dados por que estamos definindo
# uma ordem para gerar numeros aleatórios
SEED = 20

uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
# Ler arquivo csv de uma uri ou path
dados = pd.read_csv(uri)

# exibir por padrão as 5 primeiras linhas
# print(dados.head())

# Para trabalhar com as colunas em portugues, podemos realizar um map da colunar
mapa = {
    'home': 'principal',
    'how_it_works': 'como_funciona',
    'contact': 'contato',
    'bought': 'comprou'
}

# para renomar as colunas de acordo com o mapa
dados = dados.rename(columns=mapa)

# Para exibir somente as colunas que queremos podemos espeficicar
# Com os dados mapeados para portugues, podemos acessar atraves das colunas em portugues
x = dados[['principal', 'como_funciona', 'contato']]
y = dados['comprou']

# O shape é a dimensão do array numpy
print(dados.shape)

# SEPARANDO DADOS MANUALMENTE
'''
# Para treinar um modelo, vamos separar os dados para um pouco para o treino e outro para o teste
# capturar 0 a 75 posição do array
treino_x = x[:75]
treino_y = y[:75]
# capturar todas posição depois da posição 75
teste_x = x[75:]
teste_y = y[75:]
print('Treinaremos com %d elementos e testaremos com %d elemtos' %
      (len(treino_x), len(teste_x)))
'''

# O Sklearn tem o pacote de split que quebra os dados x e y para dados de treino e teste.
# TEST_SIZE = Os testes vão ter 25 % dos dados
# RANDOM_STATE = recebe semente para gerar numero com base nela. Usado para sempre separar da mesma forma os dados
# STRATIFY = para que os dados sejam separado dos testes de forma estratificada,
# ou seja para que não haja pontos altos e baixos(muitas pessoas que visitou e não comprou).
# O estrato de amostras vai se basear na separação do y
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, random_state=SEED, test_size=0.25, stratify=y)
print(treino_y)
print(teste_y.value_counts())

# com os dados prontos, vamos treinar um modelo LinearSVC
modelo = LinearSVC()
# treinar modelo com os dados de treino e as etiquetas de classificação do dados de treino
modelo.fit(treino_x, treino_y)
# prever o resultado dos dados de testes
previsoes = modelo.predict(teste_x)
# saber a taixa de acerto (acuracia)
acuracia = accuracy_score(teste_y, previsoes)
print('A acuracia foi %.2f%%' % (acuracia * 100))
