import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_to_hidden = np.random.randn(hidden_size, input_size)
        self.bias_input_to_hidden = np.zeros(hidden_size)
        self.weights_hidden_to_output = np.random.randn(output_size, hidden_size)
        self.bias_hidden_to_output = np.zeros(output_size)

    def forward_propagation(self, input_data):
        mapper = np.vectorize(self.relu)
        self.hidden_raw = np.dot(self.weights_input_to_hidden, input_data) + self.bias_input_to_hidden
        # на 14 строчке переменная в которой будет применятся функция активации для self.hidden_raw для нормализации
        # данных
        self.hidden = mapper(self.hidden_raw)
        mapper = np.vectorize(self.sigmoid)
        self.output_raw = np.dot(self.weights_hidden_to_output, self.hidden) + self.bias_hidden_to_output
        # на 19 строчке переменная в которой будет применятся функция активации для self.output_raw для нормализации
        # данных
        self.output = mapper(self.output_raw)
        return self.output

    def back_propagation(self, input_data, tags, learning_rate):
        # тут можно менять функции активации если это надо
        self.error = tags - self.output
        output_error = self.error
        output_delta = output_error * self.sigmoid_backward(self.output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_to_output)
        hidden_layer_delta = hidden_layer_error * self.relu_backward(self.hidden_raw)
        output_delta = np.array([output_delta])
        hidden_data = np.array([self.hidden])

        self.weights_hidden_to_output -= output_delta.T.dot(hidden_data) * learning_rate
        self.bias_hidden_to_output -= np.sum(output_delta[0], axis=0, keepdims=True) * learning_rate
        self.weights_input_to_hidden -= hidden_layer_delta.T.dot(input_data) * learning_rate
        self.bias_input_to_hidden -= np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

    def train(self, input_data, tag, learning_rate):
        self.forward_propagation(input_data)
        self.back_propagation(input_data, tag, learning_rate)

    # def train(self, input_data, tag):


    def predict(self, input_data):
        return self.forward_propagation(input_data)

# якщо будуть якісь помилки з функціями активації, хоч дзвоніть в будь який момент з 9 ранку до 10 ночі,
# раніше ао пізніше - пишіть

# на 47 строці сігмоїдна функція активацї та на 50 - її похідна
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_backward(self, s):
        sig = self.sigmoid(s)
        return sig * (1 - sig)

# на 55 строці функція активацї ReLU та на 58 - її похідна
    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    # на 64 строці функція активацї tanh та на 67 - її похідна
    def tanh(self, t):
        return np.tanh(t)

    def tanh_backward(self, t):
        dT = self.tanh(t)
        return 1 - np.square(dT)

    def categorical_crossentropy(self, y_true, y_pred):
        loss = 0.0
        for i in range(len(y_true)):
            if y_pred[i] > 0:
                loss += -y_true[i] * np.log(y_pred[i])
        return loss

def mix_and_separate_data(data):
    n = data.shape[0]
    mixed_data = data.sample(n)
    n_train = round(n * 0.7)
    train_data = mixed_data.iloc[:n_train]
    test_data = mixed_data.iloc[n_train:n]
    return {'train': train_data, 'test': test_data}


def assign_class_number(data, column):
    unique_values = data[column].unique()
    char_to_number = {}
    for i, char in enumerate(unique_values):
        char_to_number[char] = i
    data[column] = data[column].apply(lambda x: char_to_number[x])
    return data


def replace_na_value(data, column):
    average_value = data[column].mean(skipna=True)
    data[column].fillna(value=average_value, inplace=True)
    return data


iris_data = pd.read_csv('iris.data', header=None)
iris_data = iris_data.rename(columns={0: "sepal length", 1: "sepal width", 2: "petal length", 3: "petal width", 4: "class"})
iris_data = assign_class_number(iris_data, "class")
mix_iris_data = mix_and_separate_data(iris_data)
train_iris_data = mix_iris_data["train"]
train_iris_output = train_iris_data["class"]
train_iris_data = train_iris_data.drop("class", axis=1)
test_iris_data = mix_iris_data["test"]
test_iris_output = test_iris_data["class"]
test_iris_data = test_iris_data.drop("class", axis=1)


epochs = 1000
learning_rate = 0.07
iris_network = NeuralNetwork(4, 4, 3)
for e in range(epochs):
    actual_outputs = []
    predict_outputs = []
    inputs = []
    train_lose = []
    for index, row in train_iris_data.iterrows():
        row = np.array(row)
        expected = np.zeros(3)
        expected[train_iris_output[index]] = 1
        actual_outputs.append(expected)
        iris_network.train(row, expected, learning_rate)
        inputs.append(row)
    for i in range(len(inputs)):
        train_lose.append(iris_network.categorical_crossentropy(actual_outputs[i], iris_network.predict(inputs[i])))
    if e % 50 == 0:
        print(np.max(train_lose))
for index, row in test_iris_data.iterrows():
    row = np.array(row)
    expected = test_iris_output[index]
    predict_output = iris_network.predict(np.array(row))
    print(predict_output)
    print(expected)
    # print("For input: " + row + " the prediction is: " + predict_output + ", expected: " + expected)
