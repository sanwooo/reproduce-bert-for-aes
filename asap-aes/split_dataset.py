import os 
import codecs
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def read_dataset(path):
    df = pd.read_csv(path, sep='\t')
    return df

def kfold_split(df, n_splits=5, random_state=42):
    """
        df: pd.DataFrame
    """
    for pid in [1, 2, 3, 4, 5, 6, 7, 8]:
        test_mask = df["essay_set"]==pid
        df_test = df.loc[test_mask]
        df_train_dev = df.loc[~test_mask]
        # define a k-fold split tool
        kf = KFold(n_splits, shuffle=True, random_state=random_state)
        for fid, (train_idx, dev_idx) in enumerate(kf.split(df_train_dev)):
            df_train = df_train_dev.iloc[train_idx]
            df_dev = df_train_dev.iloc[dev_idx]

            # save dataframes
            save_folder = f"bert/asap-aes/prompt_{pid}/fold_{fid}/"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            df_train.to_csv(os.path.join(save_folder, 'train.tsv'), sep='\t', index=None)
            df_dev.to_csv(os.path.join(save_folder, 'dev.tsv'), sep='\t', index=None)
            df_test.to_csv(os.path.join(save_folder, 'test.tsv'), sep='\t', index=None)
    
    return


if __name__ == '__main__':
    path = "/home/lishanyu/projects/bert/asap-aes/training_set_rel3.tsv"
    df = read_dataset(path)
    kfold_split(df, n_splits=5)