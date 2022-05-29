from core.datasets.gesture_dataset import GestureDataset


module_dict = {
    'GestureDataset': GestureDataset,
}


def get_dataset(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown dataset: %s' % name)
    else:
        return obj
