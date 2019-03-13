from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# classificação, ele tem pelos longos? ele tem perna curta? ela faz au au(late)?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# porco = 1, cachorro = 0
# Classes 0 e 1 que identifica porco e cachorro.
treino_y = [1, 1, 1, 0, 0, 0] # conhecido como labels / etiquetas

# Treinando um modelo com o LinearSVC
model = LinearSVC()
# O fit vai treinar um modelo com os dados e a classes que esses dados correspode.
# Isso é aprendizado supervisionado.
model.fit(treino_x, treino_y)

# Identificar esses dados de acordo com o modelo treinado
misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

teste_x = [misterio1, misterio2, misterio3]
# Teste de acertos. Os valores do array segnifica 0 = errou e 1 = acertou. Vamos testar a quantidade de acertos
# que o modelo acertou com base nos testes.
teste_y = [0, 1, 1] # labels


# Prever o resultado
# O sklearn usa como base de calculo a biblioteca cientifica numpy
# O retorno de predict é um numpy.array
previsoes = model.predict(teste_x)

corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_de_acerto = (corretos / total) * 100
print(f'taxa de acerto %.2f' % taxa_de_acerto )

# O sklearn tem uma função que mede acuracia do modelo.
# Primeiro parametros são os resultados verdadeiros e o segundo as previsões
accuracy_score = accuracy_score(teste_y, previsoes)
print(f'taxa de acerto %.2f' % (taxa_de_acerto * 100) )
