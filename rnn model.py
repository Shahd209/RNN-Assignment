import tkinter as tk
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, inputs, h_prev):
        self.h_prev = h_prev
        self.h = np.tanh(np.dot(self.Wxh, inputs) + np.dot(self.Whh, h_prev) + self.bh)
        self.y = np.dot(self.Why, self.h) + self.by
        return self.y, self.h

    def backward(self, inputs, h_prev, d_y):
        d_Why = np.dot(d_y, self.h.T)
        d_by = np.sum(d_y, axis=1, keepdims=True)
        d_h = np.dot(self.Why.T, d_y) * (1 - self.h ** 2)
        d_Wxh = np.dot(d_h, inputs.T)
        d_Whh = np.dot(d_h, h_prev.T)
        d_bh = np.sum(d_h, axis=1, keepdims=True)
        return d_Wxh, d_Whh, d_Why, d_bh, d_by, d_h

    def update_weights(self, d_Wxh, d_Whh, d_Why, d_bh, d_by, learning_rate=0.01):
        self.Wxh -= learning_rate * d_Wxh
        self.Whh -= learning_rate * d_Whh
        self.Why -= learning_rate * d_Why
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

text = ["I", "am", "learning", "machine"]
vocab = list(set(text))
word_to_idx = {word.lower(): i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

input_sequence = [word_to_idx[word.lower()] for word in text[:-1]]
target_word = word_to_idx[text[-1].lower()]

rnn = SimpleRNN(input_size=len(vocab), hidden_size=5, output_size=len(vocab))
epochs = 1000
learning_rate = 0.1
h_prev = np.zeros((5, 1))

for epoch in range(epochs):
    inputs = np.zeros((len(vocab), 1))
    for i in input_sequence:
        inputs[i] = 1
    target = np.zeros((len(vocab), 1))
    target[target_word] = 1
    y, h_prev = rnn.forward(inputs, h_prev)
    d_y = y - target
    d_Wxh, d_Whh, d_Why, d_bh, d_by, d_h = rnn.backward(inputs, h_prev, d_y)
    rnn.update_weights(d_Wxh, d_Whh, d_Why, d_bh, d_by, learning_rate)

def predict():
    input_words = entry.get().lower().split()
    if input_words != ["i", "am", "learning"]:
        result_label.config(text="You must input exactly: 'I am learning'")
        return
    inputs = np.zeros((len(vocab), 1))
    for word in input_words:
        idx = word_to_idx.get(word, -1)
        if idx == -1:
            result_label.config(text="Word not in vocabulary.")
            return
        inputs[idx] = 1

    y, _ = rnn.forward(inputs, np.zeros((5, 1)))
    pred_idx = np.argmax(y)
    predicted_word = idx_to_word[pred_idx]
    result_label.config(text=f"Predicted 4th word: {predicted_word}")

root = tk.Tk()
root.title("RNN Word Predictor")

label = tk.Label(root, text="Enter exactly: I am learning")
label.pack()

entry = tk.Entry(root)
entry.pack()

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack()

result_label = tk.Label(root, text="Prediction will appear here.")
result_label.pack()

root.mainloop()
