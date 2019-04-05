'''
PROJETO 2, VAMOS ANALISAR OS DADOS GERADO ATRAVES DE HORAS DE PROJETOS, VALORES E PROJETOS QUE FORAM FINALIZADOS.
COM ISSO VAMOS PREVER SE A HORA E O VALOR DO PROJETO VAI TER COMO RESULTADO O PROJETO FINALIDO
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
dados = pd.read_csv(uri)

# mapear colunas para o portugues
a_renomear = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizado'
}

dados = dados.rename(columns=a_renomear)

# Para facilitar o entendimento dos dados, em vez de não finalizado, vamos trocar para finalizados e os valores invertidos.
troca = {
    1: 0,
    0: 1
}
# criar uma nova coluna e adicionar o acontrario de não finalizado que é finalizado
dados['finalizado'] = dados.nao_finalizado.map(troca)
# print(dados.tail()) listar os 5 ultimos registro. tail = rabo
# print(dados.head()) listar os 5 primeiros registro. head = cabeça

# criar um grafico de frequencia onde o x são as horas e o y os preços e os date = dados
# podemos definir cores em cada pronto pelo hue com base no finalizado
# sns.scatterplot(x='horas_esperadas', y='preco', hue='finalizado', data=dados)

# podemos usar o rel plot para plotar dois grafico com duas colunas com base nos finalizado e não finalizados
# sns.relplot(x='horas_esperadas', y='preco',
#             col='finalizado',  hue='finalizado', data=dados)

# # exibir janela com o grafico
# plt.show()

# separando dados para treinar um modelo para prever
# se a quantidade de horas e o valor a pagar vai finalizar dar para finalizar o projeto
x = dados[['horas_esperadas', 'preco']]
y = dados[['finalizado']]

# O train_test_split separa dos dados com uma semente randomica. Para isso vamos definir uma semente para que
# o resultado seja sempre o mesmo, ou seja ele sempre vai separar da mesma forma os dados por que estamos definindo
# uma ordem para gerar numeros aleatórios
SEED = 20

# O Sklearn tem o pacote de split que quebra os dados x e y para dados de treino e teste.
# TEST_SIZE = Os testes vão ter 25 % dos dados
# RANDOM_STATE = recebe semente para gerar numero com base nela. Usado para sempre separar da mesma forma os dados
# STRATIFY = para que os dados sejam separado dos testes de forma estratificada,
# ou seja para que não haja pontos altos e baixos(muitas pessoas que visitou e não comprou).
# O estrato de amostras vai se basear na separação do y
treino_x, teste_x, treino_y, teste_y = train_test_split(
    x, y, random_state=SEED, test_size=0.25, stratify=y)

print('Treinaremos com %d elementos e testaremos com %d elemtos' %
      (len(treino_x), len(teste_x)))

# com os dados prontos, vamos treinar um modelo LinearSVC
modelo = LinearSVC()
# treinar modelo com os dados de treino e as etiquetas de classificação do dados de treino
modelo.fit(treino_x, treino_y)
# prever o resultado dos dados de testes
previsoes = modelo.predict(teste_x)
# saber a taixa de acerto (acuracia)
acuracia = accuracy_score(teste_y, previsoes)
print('A acuracia foi %.2f%%' % (acuracia * 100))

# O algortmo de base é importante para saber se o modelo esta ruim ou bom, pelo fato de testar
# um dados de um unico valor.
# gerar um arrar com 540 posições de valor 1
previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base)
print('A acuracia do algoritmo de baseline foi %.2f%%' % (acuracia * 100))

# Para analisar porque a acursacia do modelo esta ruim, vamos testar o modelo de acordo com o grafico gerado

# sns.scatterplot(x=teste_x.horas, y='preco', hue='finalizado', data=dados)

# plt.show()

# temos que min e max de teste_x e y. Na posição x = horas_esperadas e y = preco
x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

print(x_min, x_max, y_min, y_max)

# Para marcar no grafico os pontos do treinamento, vamos definir um tamanho de pixel
pixels = 100
# No eixo x, entre o minimo e maximo, de quanto a quantos vamos ter pontos de pixels
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

# Com o eixo x e eixo y pronto, vamos mesclar em um grid, porem o meshgrid somente cria 100x100 repedindo 100 vezes
xx, yy = np.meshgrid(eixo_x, eixo_x)
# vamos concatenar o x com o y criando pontos validos no plano cartesiano
pontos = np.c_[xx.ravel(), yy.ravel()]
print(pontos)

# com todos os prontos prontas, podemos prever com o modelo os resultados desses pontos
Z = modelo.predict(pontos)
# porem a previsão gerada pelo modelo tem a dimenção (10000,) e nosso xx e yy é 100x100. Vamos redimencionar 10000 para 100x100
Z = np.reshape(Z, (100,100))
print(Z.shape)

# Vamos desenha no grafico a "curva" de acertos e erros
# alpha é a transparencia
print(xx, yy)
plt.contourf(xx, yy, Z, alpha=0.3)

# Vamos plotar os resultados da previsão. O scatter vai plotar a dispersão dos pontos
# c = cor, pore exemplo c="blue", ou uma sequencia de valores 0 ou 1.
# s = size, tamanho dos pontos
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y.finalizado, s=1)

# exibir janela com o grafico
plt.show()

