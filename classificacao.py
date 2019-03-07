from sklearn.svm import LinearSVC

# classificação, ele tem pelos longos? ele tem perna curta? ela faz au au(late)?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# porco = 1, cachorro = 0
# Classes 0 e 1 que identifica porco e cachorro.
classes = [1, 1, 1, 0, 0, 0]

# Treinando um modelo com o LinearSVC
model = LinearSVC()
# O fit vai treinar um modelo com os dados e a classes que esses dados correspode.
# Isso é aprendizado supervisionado.
model.fit(dados, classes)

# Identificar esses dados de acordo com o modelo treinado
misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

testes = [misterio1, misterio2, misterio3]

# Prever o resultado
previsoes = model.predict(testes)

testes_classes = [0,1,1]