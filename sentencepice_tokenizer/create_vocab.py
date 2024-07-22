from SentencePiece import SentencePiece , get_spm_tokens
import sentencepiece as spm
import toml
tokenizer = SentencePiece(
        model_dir="wordpiece/pbe/300",
         vocab_size=300,
       annotation_train="./corpus.csv",
        annotation_read="text",
         model_type="bpe",
         character_coverage=1.0,
     )