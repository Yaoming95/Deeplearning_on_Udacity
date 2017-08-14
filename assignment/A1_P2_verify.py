from A1_3_convert2Array import *



def data_verify(datasets = None):
    if datasets is None:
        [datasets, _] = get_datasets()
        pickle_file = datasets[0]  # index 0 should be all As, 1 = all Bs, etc.
    with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)  # unpickle
        sample_idx = np.random.randint(len(letter_set))  # pick a random image index
        sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
        plt.figure()
        plt.imshow(sample_image)  # display it
        pass

if __name__ == '__main__':
    data_verify()