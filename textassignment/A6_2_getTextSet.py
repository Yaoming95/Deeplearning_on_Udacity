
from textassignment.A6_1_readData import read_data


def get_set():
    text = read_data()
    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    # train_size = len(train_text)
    # print(train_size, train_text[:64])
    # print(valid_size, valid_text[:64])
    return train_text, valid_text
