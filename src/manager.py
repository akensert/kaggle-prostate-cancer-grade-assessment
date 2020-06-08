import pandas as pd
import numpy as np
import os


class DataManager:
    """This class is created only for convenience --- and is specific to
    the PANDA [Kaggle challenge] dataset. It can directly be used locally,
    on a server, or in a Kaggle kernel

    Example usage:
        train_data, test_data, sub_data = DataManager.get_all_files()
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
    def get_train_file(cls):
        return pd.read_csv(cls.get_path() + 'train.csv')

    @classmethod
    def get_test_file(cls):
        return pd.read_csv(cls.get_path() + 'test.csv')

    @classmethod
    def get_submission_file(cls):
        return pd.read_csv(cls.get_path() + 'sample_submission.csv')

    @classmethod
    def get_all_files(cls):
        return (
            cls.get_train_file(),
            cls.get_test_file(),
            cls.get_submission_file()
        )

if __name__ == "__main__":
    train_data, test_data, sub_data = DataManager.get_all_files()
    print("train_data shape   =", train_data.shape)
    print("train_data columns =", train_data.columns.values)
    print("train_data dtypes  =", train_data.dtypes.values)
    print("test_data shape    =", test_data.shape)
    print("test_data columns  =", test_data.columns.values)
    print("test_data dtypes   =", test_data.dtypes.values)
    print("sub_data shape     =", sub_data.shape)
    print("sub_data columns   =", sub_data.columns.values)
    print("sub_data dtypes    =", sub_data.dtypes.values)
