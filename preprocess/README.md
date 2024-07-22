### Run process dataset

1. Start data preparation for Wenetspeech:
        ```
        python preprocess.py --config config.toml
        ```
2. Please provide path parameters in file config.toml
    
3. If you are using Ubuntu OS, please change the code in line 73 to 
        ```
        full_path = find_element_containing_a(path, opus_file_paths)
        ```
4. On Windows OS , just run the code .

### Convert into pinyin style


1. Please provide path parameters in file config.toml : 
        dict_save_path = "" for the pickle mapping file

        input_pinyin = ""  the input folder to convert

        E.g : in default config file
2. Start convert task for Wenetspeech:
        ```
        python to_pinyin.py --config config.toml
        ```
    
3. To delete all the pinyin mapping file , run 
        ```
        python delete_relative.py --config config.toml
        ```
