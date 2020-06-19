import numpy as np
import pandas as pd
import os
from sklearn import model_selection


class DataManager:
    """This class is created only for convenience --- and is specific to
    the PANDA [Kaggle challenge] dataset. It can directly be used locally,
    on a server, or in a Kaggle kernel

    Example usage:
        train_data, test_data, sub_data = DataManager.get_all_data()
    """
    @staticmethod
    def get_path():
        directory = 'input/prostate-cancer-grade-assessment/'
        navigate = ''
        while 1:
            if os.path.isdir(navigate + directory):
                path = navigate + directory
                break
            navigate += '../'
            if len(navigate) > 15:
                print("Directory {} not found".format(directory))
                path = None
                break
        return path

    @classmethod
    def get_train_data(cls, split=False, test_size=0.2, random_state=42, add_image_size_info=False):
        if os.path.isfile(cls.get_path() + 'train.csv'):
            data = pd.read_csv(cls.get_path() + 'train.csv')
            if add_image_size_info:
                image_size = {}
                for f in os.listdir(cls.get_path() + 'train_images/'):
                    fp = os.path.join(cls.get_path() + 'train_images/', f)
                    image_size[f[:-5]] = round(os.path.getsize(fp) * (1/1_000_000), 3)
                data['image_size'] = data.image_id.map(image_size)
            if split:
                return _advanced_split(data, test_size, random_state)
            else:
                return data
        return None

    @classmethod
    def get_test_data(cls):
        if os.path.isfile(cls.get_path() + 'test.csv'):
            data = pd.read_csv(cls.get_path() + 'test.csv')
            data['isup_grade'] = 1
            data['gleason_score'] = 1
            return data
        return None

    @classmethod
    def get_submission_data(cls):
        if os.path.isfile(cls.get_path() + 'sample_submission.csv'):
            return pd.read_csv(cls.get_path() + 'sample_submission.csv')
        return None

    @classmethod
    def get_all_data(cls):
        return (
            cls.get_train_data(),
            cls.get_test_data(),
            cls.get_submission_data()
        )


def _advanced_split(data, test_size=0.2, random_state=42):
    """Splits the data into train and valid set, in a stratified
    manner. Also puts similar examples in the same set (based
    on _/input/similar_examples_hashXXX.npy)"""

    data['fold'] = -1

    classes = (
        np.where(data.data_provider == 'karolinska', 6, 0)
        + data.isup_grade.values)

    skf = model_selection.StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state)

    skf_iterator = skf.split(
        X=data.image_id,
        y=classes)

    train_idx, valid_idx = next(skf_iterator)
    data.loc[valid_idx, 'fold'] = 0
    data.loc[train_idx, 'fold'] = 1

    similar = np.load('input_/similar_examples_hash094.npy')

    assign_fold = {0: [], 1: []}
    for xid, yid in similar:
        xid_fold = data.loc[xid, 'fold']
        if xid not in assign_fold[0] and xid not in assign_fold[1]:
            assign_fold[xid_fold].append(xid)
        if yid not in assign_fold[0] and yid not in assign_fold[1]:
            assign_fold[xid_fold].append(yid)

    for k, v in assign_fold.items():
        data.loc[v, 'fold'] = k

    return data[data.fold != 0], data[data.fold == 0]


if __name__ == "__main__":

    train_data, test_data, sub_data = DataManager.get_all_data()

    if train_data is not None:
        print("train_data shape   =", train_data.shape)
        print("train_data columns =", train_data.columns.values)
        print("train_data dtypes  =", train_data.dtypes.values)
    else:
        print("train.csv not found")

    if test_data is not None:
        print("test_data shape    =", test_data.shape)
        print("test_data columns  =", test_data.columns.values)
        print("test_data dtypes   =", test_data.dtypes.values)
    else:
        print("test.csv not found")

    if sub_data is not None:
        print("sub_data shape     =", sub_data.shape)
        print("sub_data columns   =", sub_data.columns.values)
        print("sub_data dtypes    =", sub_data.dtypes.values)
    else:
        print("submission.csv not found")
