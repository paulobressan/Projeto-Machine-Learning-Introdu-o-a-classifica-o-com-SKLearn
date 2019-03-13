'''
PROJETO 2, VAMOS ANALISAR OS DADOS GERADO ATRAVES DA NAVEGAÇÃO DE UM CLIENTE EM UM SITE
'''

# Bilbioteca de analise de dados
import pandas as pd
# sklearn para treinar modelos
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

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
    'bought':'comprou'
}

# para renomar as colunas de acordo com o mapa
dados = dados.rename(columns = mapa)

# Para exibir somente as colunas que queremos podemos espeficicar 
# Com os dados mapeados para portugues, podemos acessar atraves das colunas em portugues 
x = dados[['principal', 'como_funciona', 'contato']]
y = dados['comprou']

# O shape é a dimensão do array numpy
print(dados.shape)

# Para treinar um modelo, vamos separar os dados para um pouco para o treino e outro para o teste
# capturar 0 a 75 posição do array
treino_x = x[:75]
treino_y = y[:75]
# capturar todas posição depois da posição 75
teste_x = x[75:]
teste_y = y[75:]
print('Treinaremos com %d elementos e testaremos com %d elemtos' % (len(treino_x), len(teste_x)))

# com os dados prontos, vamos treinar um modelo LinearSVC
modelo = LinearSVC()
# treinar modelo com os dados de treino e as etiquetas de classificação do dados de treino
modelo.fit(treino_x, treino_y)
# prever o resultado dos dados de testes
previsoes = modelo.predict(teste_x)
# saber a taixa de acerto (acuracia)
acuracia = accuracy_score(teste_y, previsoes)
print('A acuracia foi %.2f%%' % (acuracia * 100))