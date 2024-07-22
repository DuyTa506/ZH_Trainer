import os.path as osp
import numpy as np
import os
import re
import toml
from tqdm import tqdm
import argparse
import os
import numpy as np
from tqdm import tqdm

def mkdir(path):
    if not osp.exists(path):
        os.makedirs(path)


def savenp(dir,name,a):
    mkdir(dir)
    np.save(osp.join(dir,name),a)
    
import pickle
def load_word_to_phone_seq(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def chinese_to_pinyin(text, word_to_pinyin):
    result = []
    for char in text:
        pinyin = word_to_pinyin.get(char, char)
        pinyin = pinyin.replace(' ', '')
        result.append(pinyin)
    return ' '.join(result)





def split_line(line, max_length=512):
    chunks = [line[i:i+max_length] for i in range(0, len(line), max_length)]
    return chunks

def translate(input, path, word_to_pinyin):
    output = input.copy()
    input_path = path
    print(path)
    file = open(input_path, 'w', encoding="utf-8")
    for k in range(len(output)):
        i = 0
        while i < len(output[k]):
            if re.search('[\u4e00-\u9fff]', output[k][i]):
                char = output[k][i]

                py = chinese_to_pinyin(char, word_to_pinyin)
                
                output[k] = output[k][:i] + py + ' ' + output[k][i+1:]
                i += len(py)
            else:
                i += 1
        file.write("%s\n" % output[k])
    file.close()
    return output

def add_suffix_to_txt_files(file_name):
    base_name = os.path.basename(file_name)
    new_name = os.path.splitext(base_name)[0] + "_dacidian_pinyin.txt"
    new_path = os.path.join(os.path.dirname(file_name), new_name)
    return new_path

def get_all_txt_files(root_folder):
    if not os.path.exists(root_folder):
        print(f"Thư mục '{root_folder}' không tồn tại.")
        return []
    txt_files = [os.path.join(root, file) for root, dirs, files in os.walk(root_folder) for file in files if file.endswith('.txt')]
    return txt_files


def convert_pinyin(input_folders, word_to_pinyin):
    all_txt_files = []
    for folder in input_folders.split(','):
        folder = folder.strip()
        txt_files = get_all_txt_files(folder)
        all_txt_files.extend(txt_files)
    
    total_files = len(all_txt_files)
    files_processed = 0
    
    print(f"Total files to process: {total_files}")

    for file_path in tqdm(all_txt_files, desc="Processing files"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.readlines()
        
        
        translated_content = translate(content, file_path, word_to_pinyin)
        
        new_path = add_suffix_to_txt_files(file_path)
        
        with open(new_path, 'w', encoding='utf-8') as new_file:
            new_file.write('\n'.join(translated_content))
        
        files_processed += 1
        print(f"Processed file {files_processed}/{total_files}: {file_path} -> {new_path}")
    
    print("Conversion completed.")
    
def main():
    parser = argparse.ArgumentParser(description="Convert to pinyin")
    parser.add_argument('--config', type=str, default='config.toml', help="Path to the TOML configuration file")
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = toml.load(f)
    dict_save_path = config['paths']['dict_save_path']
    input_pinyin = config['paths']['input_pinyin']
     
    word_to_phone_seq_loaded = load_word_to_phone_seq(dict_save_path)
    
    convert_pinyin(input_pinyin, word_to_phone_seq_loaded)


if __name__ == "__main__":
    main()
    
    

    
