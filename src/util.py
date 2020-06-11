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
    def get_train_data(cls, split_fold=None):
        if os.path.isfile(cls.get_path() + 'train.csv'):

            data = pd.read_csv(cls.get_path() + 'train.csv')

            if split_fold is not None:
                classes = (
                    np.where(data.data_provider == 'karolinska', 6, 0)
                    + data.isup_grade.values)

                skf = model_selection.StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42).split(data.image_id, y=data.isup_grade)

                data['fold'] = -1
                for i, (_, valid_idx) in enumerate(skf):
                    data.loc[valid_idx, 'fold'] = i

                return data[data.fold != split_fold], data[data.fold == split_fold]
            else:
                return data
        return None

    @classmethod
    def get_test_data(cls):
        if os.path.isfile(cls.get_path() + 'test.csv'):
            return pd.read_csv(cls.get_path() + 'test.csv')
        return None

    @classmethod
    def get_submission_data(cls):
        if os.path.isfile(cls.get_path() + 'sample_submission.csv'):
            return pd.read_csv(cls.get_path() + 'sample_submission.csv')
        return None

    @classmethod
    def get_all_data(cls):
        return (
            cls.get_train_file(),
            cls.get_test_file(),
            cls.get_submission_file()
        )

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
