import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Carregar os dados do arquivo
data = np.loadtxt('data_tp1', delimiter=',')

# Separar os dados de entrada (features) e os rótulos (targets)
features = data[:, :-1]
targets = data[:, -1]

# Definir os parâmetros da rede neural
num_input = features.shape[1]  # Número de atributos de entrada
num_hidden = 10  # Número de neurônios na camada oculta
num_output = 1  # Número de neurônios na camada de saída

# Definir os valores de "learning_rate" a serem testados
learning_rates = [0.001, 0.01, 0.1, 1.0]

# Inicializar listas para armazenar os valores de perda média de cada "learning_rate"
losses = [[] for _ in range(len(learning_rates))]

# Iniciar a sessão do TensorFlow
with tf.Session() as sess:
    for i, learning_rate in enumerate(learning_rates):
        tf.set_random_seed(42)  # Definir semente aleatória para reprodução dos resultados

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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

        # Inicializar as variáveis globais
        init = tf.global_variables_initializer()
        sess.run(init)

        # Definir o número de épocas e o tamanho do lote (batch size)
        num_epochs = 1000
        batch_size = 32

        # Treinar a rede neural
        for epoch in range(num_epochs):
            # Embaralhar os dados de treinamento
            indices = np.random.permutation(len(features))
            features_shuffled = features[indices]
            targets_shuffled = targets[indices]

            # Dividir os dados em lotes (batches)
            num_batches = len(features) // batch_size
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                batch_x = features_shuffled[start:end]
                batch_y = targets_shuffled[start:end]

                # Executar o passo de treinamento (forward pass e backpropagation)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            # Calcular a perda média da época atual
            avg_loss = sess.run(loss, feed_dict={X: features, Y: targets})
            losses[i].append(avg_loss)

            # Imprimir a perda média a cada 100 épocas
            if (epoch + 1) % 100 == 0:
                print("Época:", epoch + 1, "Learning Rate:", learning_rate, "Perda média:", avg_loss)

# Plotar os gráficos de perda média para cada "learning_rate"
for i, learning_rate in enumerate(learning_rates):
    plt.plot(losses[i], label='Learning Rate: ' + str(learning_rate))

plt.xlabel('Épocas')
plt.ylabel('Perda média')
plt.legend()
plt.show()
