#!/usr/bin/python

from numpy import exp, array, random, dot

class PerceptronNeuralNetwork:

    # CONSTRUTOR
    def __init__(self):
        self.peso = random.random((3, 1))
        self.error = 0



    def treinar(self, inputs, outputs, num):
        for iteration in range(num):

            # Testa sempre os mesmos inputs, multiplicados por pesos 
            # cada vez mais regulado conforme o esperado
            output = self.testar(inputs)

            # Calcula o erro
            self.error = outputs - output

            # Ajusta o peso
            adjustment = dot(inputs.T, self.error * output * (1 - output))
            self.peso += adjustment



    def testar(self, inputs):
        # Calcula 1 output para cada input
        resultado = dot(inputs, self.peso)
        
        # Usa a funcao de ativacao sigmoid para normalizar output
        resultado_normalizado = self.sigmoide(resultado);
        return resultado_normalizado



    def sigmoide(self, x):
        return 1 / (1 + exp(-x))

# ====================================================================

network = PerceptronNeuralNetwork()

# TREINA

inputs = array([[1, 1, 1],
                [1, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [0, 0, 0],
                [0, 0, 1]])

outputs = array([[1],
                 [1],
                 [0],
                 [1],
                 [0],
                 [0]])

network.treinar(inputs, outputs, 1000)

# (+ treino && + dados) => peso+ajustado
# peso+ajustado => - erro

# Um erro pra cada input
print("Erro: \n%s\n" % network.error)

# Um peso pra cada int do input
print("Peso: \n%s\n" % network.peso)

# PENSA

output = network.testar(array([1, 1, 0]))
print("Input: [1, 1, 0]")
print("Resposta: %s\n\n" % output) # Output esperado == 1

output = network.testar(array([0, 1, 1]))
print("Input: [0, 1, 0]")
print("Resposta: %s" % output) # Output esperado == 0
