import os
import string
import pandas as pd
import sys
import re
import librosa
import numpy as np
from pandarallel import pandarallel
from typing import Dict, List
import dask.dataframe as dd
# For testing 
sys.path.append('..')
from dask import delayed , compute
from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset


class BaseDataset(Dataset):
    def __init__(self, rank, dist, sr,  special_tokens, init_pq = ""):
        self.rank = rank
        self.dist = dist
        self.sr = sr
        self.init_pq = init_pq
        # Special characters to remove in your data 

        print("Load dataframe for train dataset")
        if os.path.isfile(os.path.join(self.init_pq, "train.csv")):
            print("Found train data file !!")
            self.df = self.load_pq_file(os.path.join(self.init_pq, "train.parquet"))
        else:
                raise ValueError("Cannot read train data file , the file {} not exist".format(os.path.join(self.init_pq, "train.csv")))

        self.special_tokens = special_tokens
        
    def get_vocab_dict(self) -> Dict[int, str]:
        all_text = " ".join(list(self.df["transcript"]))
        #  remove special tokens in all_text, otherwise it will tokenize the special tokens' characters. Eg: <unk> -> '<', 'u', 'n', 'k', '>'
        for v in self.special_tokens.values():
            all_text = all_text.replace(v, '')
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        for v in self.special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        return vocab_dict

        
    def load_pq_file(self, data_path) :
        df = pd.read_parquet(data_path)
        print("Readed old data file , begin filtering !!!")
        print(len(df))
        return df

    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.df, self.sr, self.preload_data, self.transform)
        return ds


if __name__ == '__main__':
    ds = BaseDataset(
        path = '/home/pvanh/data/zh_stt/ASR-RAMC-BIGCCSC', 
        sr = 16000, 
        preload_data = False,
        rank = 1,
        dist=None,
        delimiter="|",
        special_tokens=None, 
        # val_size = None, 
        transform = None)
    df= ds.load_data()
    print(df.head)
    vocab_dict = ds.get_vocab_dict()
    for k, v in vocab_dict.items():
        print(f'{k} - {v}')
