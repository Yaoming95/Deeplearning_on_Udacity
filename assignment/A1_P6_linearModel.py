from sklearn.linear_model import LogisticRegression
from assignment.A1_4_splitting import data_split
from sklearn.metrics import mean_squared_error, r2_score
from assignment.A1_P4_shuffling import randomize


TRAINING_NUM = 800

def mat2array(dataset):
    return dataset.reshape((dataset.shape[0], -1), order='F')

[valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels] = data_split()


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


train_dataset = train_dataset[0:TRAINING_NUM, :, :]
train_dataset = mat2array(train_dataset)
train_labels = train_labels[0:TRAINING_NUM]

valid_dataset = mat2array(valid_dataset)


logisticRegression = LogisticRegression()
logisticRegression.fit(train_dataset, train_labels)
print ('Variance score: %.2f' % logisticRegression.score(valid_dataset, valid_labels))

pass
