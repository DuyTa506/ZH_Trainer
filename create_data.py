import os
import string
import pandas as pd
import sys
import re
import librosa
import numpy as np
from pandarallel import pandarallel

import dask.dataframe as dd
# For testing 
sys.path.append('..')
from dask import delayed , compute
import toml
import argparse


class Dataset_Builder():
    def __init__(self, path, min_duration = -np.inf, max_duration = np.inf, nb_workers = 4, volume = [], init_pq = "", model_type= "pinyin", token_min = 0, token_max = 500, special_tokens= {}):
        self.volume = volume
        self.model_type = model_type
        self.init_pq = init_pq
        self.nb_workers = nb_workers
        # Special characters to remove in your data 

        self.chars_to_ignore = u"[！？。，＂＃％＆＇（）－／：；＜＝＞＠＼＾｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"  + string.punctuation.replace("_", "").replace("$", "") + ']+'  # Remove $ from chars_to_ignore

        self.label  = ["[+]", "[++]", "[*]", "[SONANT]", "[MUSIC]", "[LAUGHTER]", "[ENS]", "[SYSTEM]"]
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.df = self.create_data(path)        
    def remove_special_characters(self, transcript) -> str:
        rule = re.compile(self.chars_to_ignore)
        label_pattern = '|'.join(map(re.escape, self.label))
        rule_label = re.compile(f'(\[{label_pattern}\])')

        transcript = re.sub(rule_label, "", transcript)    

        transcript = re.sub(rule, " ", transcript).lower()
        
        transcript = ' '.join(transcript.split())
        return transcript



    def create_data(self, input_folders , batch_size=100000) -> dd.DataFrame:
            print("Training with {}".format(self.model_type))
            
            @delayed
            def _load_data_batch(input_folder):
                all_paths = []
                all_transcripts = []
                if os.path.isdir(input_folder):
                    subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
                    for subdir in subdirs:
                        subdir_path = os.path.join(input_folder, subdir)
                        wav_files = [f for f in os.listdir(subdir_path) if f.endswith(".wav")]
                        for wav_file in wav_files:
                            wav_path = os.path.join(subdir_path, wav_file)
                            if self.model_type == "pinyin":
                                transcript_path = os.path.join(subdir_path, os.path.splitext(wav_file)[0] + "_dacidian_pinyin.txt")
                            else:
                                transcript_path = os.path.join(subdir_path, os.path.splitext(wav_file)[0] + ".txt")
                            if os.path.exists(transcript_path):
                                with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
                                    transcript = transcript_file.read()
                                all_paths.append(wav_path)
                                all_transcripts.append(transcript)
                                #all_durations.append(librosa.get_duration(filename=wav_path)) #TODO fixing by append duration before create [pd] dataframe ?
                                if len(all_paths) >= batch_size:
                                    yield pd.DataFrame({
                                        'path': all_paths,
                                        'transcript': all_transcripts,

                                    })
                                    all_paths = []
                                    all_transcripts = []
                if all_paths:
                    yield pd.DataFrame({
                        'path': all_paths,
                        'transcript': all_transcripts,
                    })

            print(f"\nAdapting batch processing and saving file to {self.init_pq} ....")
            input_folders_list = input_folders.split(',')

            delayed_batches = [_load_data_batch(folder) for folder in input_folders_list]
            results = compute(*delayed_batches)
            data_frames_train = []
            for batch ,vol in zip(results, self.volume) :
                for df_batch in batch:
                    print(f"Original len : {len(df_batch)}" )
                    df_batch_train = df_batch.sample(frac=vol, random_state =42)
                    data_frames_train.append(df_batch_train)
            pd_train = pd.concat(data_frames_train, ignore_index=True)
            
            pandarallel.initialize(progress_bar=True, nb_workers = self.nb_workers)
            if self.min_duration != -np.inf or self.max_duration != np.inf : 
                print("\n*****Generate duration column*****")
                durations = pd_train['path'].parallel_apply(lambda filename: librosa.get_duration(path=filename))
                pd_train.insert(len(pd_train.columns), 'duration', durations)
                
                print("\n*****Filter out invalid audio*****")
                
                mask = (pd_train['duration'] <= self.max_duration) & (pd_train['duration'] >= self.min_duration)
                pd_last = pd_train[mask]
            pd_last['transcript'] = pd_last['transcript'].parallel_apply(self.remove_special_characters)    
            pd_last = pd_last[pd_last['transcript'] != ""].reset_index(drop=True)
            
            
            if self.init_pq != "":
                if not os.path.exists(self.init_pq):
                    os.makedirs(self.init_pq)
                pd_last.to_parquet(os.path.join(self.init_pq, 'train.parquet'), index = False)
                print(f"\nGenerated data file !")

            return pd_last 




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    args.add_argument('-c', '--config', default="config.toml", type=str,
                      help='config file path (default: None)')
    
    
    args = args.parse_args()
    config = toml.load(args.config)
    
    ds = Dataset_Builder(**config["create_data"])




