import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd

# Leitura dos dados de treinamento do arquivo
data = pd.read_csv('data_tp1.csv')
input_data = data.iloc[:, :-1].values
target_data = data.iloc[:, -1:].values

# Parâmetros da rede neural
input_size = input_data.shape[1]  # Tamanho da camada de entrada
hidden_size = 4                  # Tamanho da camada oculta
output_size = target_data.shape[1]  # Tamanho da camada de saída
learning_rate = 0.25
epochs = 10000

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + cp.exp(-x))

# Derivada da função de ativação sigmoide
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Inicialização dos pesos e viés na GPU
weights_hidden = cp.random.randn(input_size, hidden_size)
biases_hidden = cp.zeros((1, hidden_size))
weights_output = cp.random.randn(hidden_size, output_size)
biases_output = cp.zeros((1, output_size))

# Transferindo os dados para a GPU
input_data = cp.array(input_data)
target_data = cp.array(target_data)

# Treinamento da rede neural na GPU
loss_history = []
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = cp.dot(input_data, weights_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = cp.dot(hidden_layer_output, weights_output) + biases_output
    output_layer_output = sigmoid(output_layer_input)
    
    # Cálculo do erro
    error = target_data - output_layer_output
    loss = cp.mean(cp.square(error))
    loss_history.append(loss.get())
    
    # Backward pass
    d_output = error * sigmoid_derivative(output_layer_input)
    d_hidden = cp.dot(d_output, weights_output.T) * sigmoid_derivative(hidden_layer_input)
    
    # Atualização dos pesos e viés
    weights_output += learning_rate * cp.dot(hidden_layer_output.T, d_output)
    biases_output += learning_rate * cp.sum(d_output, axis=0, keepdims=True)
    weights_hidden += learning_rate * cp.dot(input_data.T, d_hidden)
    biases_hidden += learning_rate * cp.sum(d_hidden, axis=0, keepdims=True)

# Transferindo os dados de volta para a CPU para plotagem
loss_history = cp.asnumpy(cp.array(loss_history))

# Plotando a curva de aprendizado
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Previsões finais
hidden_layer_output = sigmoid(cp.dot(input_data, weights_hidden) + biases_hidden)
predictions = sigmoid(cp.dot(hidden_layer_output, weights_output) + biases_output)
print('Previsões finais:')
print(predictions.get())
