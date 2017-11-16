import string

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

valid_size = 1000

batch_size=64
num_unrollings=10

num_nodes = 64
num_steps = 7001
summary_frequency = 100
