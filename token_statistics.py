import argparse
import os
import toml
from transformers import Wav2Vec2CTCTokenizer , Wav2Vec2FeatureExtractor
import pandas as pd
from pandarallel import pandarallel
import matplotlib.pyplot as plt
import seaborn as sns

special_token = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>"
}

exacture = Wav2Vec2FeatureExtractor()
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", 
                                 **special_token,
                                 word_delimiter_token="|", do_lower_case = True)

def count_token(input_text, tokenizer):
    return len(tokenizer.encode(input_text))

def statistic_data(data_file, nb_workers):
    df = pd.read_parquet(data_file)
    pandarallel.initialize(progress_bar=True, nb_workers=nb_workers)
    print("\n*****Counting token length for each record*****")
    df['token_len'] = df['transcript'].parallel_apply(lambda transcripts: count_token(transcripts, tokenizer=tokenizer))
    
    max_token_len = df['token_len'].max()
    min_token_len = df['token_len'].min()
    
    
    max_duration = df['duration'].max()
    min_duration = df['duration'].min()
    
    print("Max token length: ", max_token_len)
    print("Max duration length: ", max_duration)
    print("--------------------------------")
    print("Min token length: ", min_token_len)
    print("Min duration length: ", min_duration)
    
    correlation = df['duration'].corr(df['token_len'])
    print("Correlation between duration and token length: ", correlation)
    
        
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration', y='token_len', data=df)
    plt.title('Relationship between Duration and Token Length')
    plt.xlabel('Duration')
    plt.ylabel('Token Length')
    output_dir = "./stats"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "statistics.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    df.to_parquet(data_file)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASR TRAIN ARGS')
    parser.add_argument('-c', '--config', default="config.toml", type=str,
                        help='config file path (default: config.toml)')
    
    args = parser.parse_args()
    config = toml.load(args.config)
    nb_workers = config["create_data"]["nb_workers"]
    data_path = os.path.join(config["create_data"]["init_pq"], "train.parquet")
    statistic_data(data_path, nb_workers)
