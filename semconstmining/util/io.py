import pickle


def write_pickle(p_map, path):
    with open(path, 'wb') as handle:
        pickle.dump(p_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as handle:
        p_map = pickle.load(handle)
        return p_map
