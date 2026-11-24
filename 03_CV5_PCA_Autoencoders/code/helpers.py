import pandas as pd

def load_file(filename = 'mnist_train.csv'):
    df = pd.read_csv(filename)
    data = df.to_numpy()
    y = data[:, 0]
    X = data[:, 1:] / 255.0
    return X, y