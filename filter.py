import argparse
import os
import numpy as np
import toml
from transformers import Wav2Vec2CTCTokenizer
import pandas as pd
from pandarallel import pandarallel





def count_token(input_text, tokenizer):
    return len(tokenizer.encode(input_text))


def filter_token(data_file, nb_workers, token_max, token_min, tokenizer):
    df = pd.read_parquet(data_file)
    pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)
    print("Total samples: ", len(df)) 
    print("\n*****Counting token length for each record*****")
    if 'token_len' not in df.columns :
        df['token_len'] = df['transcript'].parallel_apply(lambda transcripts: count_token(transcripts, tokenizer=tokenizer))
        
    if token_max != np.inf and token_min != - np.inf: 
        mask = (df['token_len'] <= token_max) & (df['token_len'] >= token_min)
        df = df[mask]
    print("Filtered samples : " ,len(df))
    df.to_parquet(data_file)
    print("Done filter datasets base on token count !")
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    parser.add_argument('-c', '--config', default="config.toml", type=str,
                        help='config file path (default: config.toml)')
    
    args = parser.parse_args()
    config = toml.load(args.config)
    nb_workers = config["create_data"]["nb_workers"]
    data_path = os.path.join(config["create_data"]["init_pq"], "train.parquet")
    
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
                                 **config["create_data"]["special_tokens"],
                                 word_delimiter_token="|", do_lower_case = True)
    
    filter_token(data_path, nb_workers,config["create_data"]["token_max"], config["create_data"]["token_min"], tokenizer)