#https://medium.com/ensina-ai/redes-neurais-com-tensorflow-primeiros-passos-20847dd5d27f
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # Quantidade de entradas
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # Quantidade de neurônios

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))
"""

import numpy as np
import tensorflow as tf

# Carregar os dados do arquivo
data = np.loadtxt('data_tp1', delimiter=',')

# Separar os dados de entrada (features) e os rótulos (targets)
features = data[:, :-1]
targets = data[:, -1]

# Definir os parâmetros da rede neural
num_input = features.shape[1]  # Número de atributos de entrada
num_hidden = 10  # Número de neurônios na camada oculta
num_output = 1  # Número de neurônios na camada de saída

# Definir os placeholders para os dados de entrada e os rótulos
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_output])

# Definir as variáveis para os pesos e os bias da rede neural
weights = {
    'hidden': tf.Variable(tf.random_normal([num_input, num_hidden])),
    'output': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden])),
    'output': tf.Variable(tf.random_normal([num_output]))
}

# Construir o modelo da rede neural
hidden_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

# Definir a função de perda e o otimizador
loss = tf.reduce_mean(tf.square(output_layer - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# Inicializar as variáveis globais
init = tf.global_variables_initializer()

# Definir o número de épocas e o tamanho do lote (batch size)
num_epochs = 1000
batch_size = 32

# Iniciar a sessão do TensorFlow
with tf.Session() as sess:
    sess.run(init)

    # Treinar a rede neural
    for epoch in range(num_epochs):
        # Embaralhar os dados de treinamento
        indices = np.random.permutation(len(features))
        features_shuffled = features[indices]
        targets_shuffled = targets[indices]

        # Dividir os dados em lotes (batches)
        num_batches = len(features) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_x = features_shuffled[start:end]
            batch_y = targets_shuffled[start:end]

            # Executar o passo de treinamento (forward pass e backpropagation)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Calcular a perda média da época atual
        avg_loss = sess.run(loss, feed_dict={X: features, Y: targets})
        print("Época:", epoch+1, "Perda média:", avg_loss)

    # Realizar previsões após o treinamento
    predictions = sess.run(output_layer, feed_dict={X: features})

    # Imprimir as previsões
    print("Previsões:")
    print(predictions)


