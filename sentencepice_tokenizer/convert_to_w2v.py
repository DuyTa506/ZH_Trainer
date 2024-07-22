from SentencePiece import SentencePiece , get_spm_tokens
import sentencepiece as spm
import toml
import json
# tokenizer = SentencePiece(
#         model_dir="wordpiece/pbe/200",
#         vocab_size=200,
#         annotation_train="./corpus.csv",
#         annotation_read="text",
#         model_type="bpe",
#         character_coverage=1.0,
#     )
sp = spm.SentencePieceProcessor(model_file='wordpiece//pbe//642_bpe.model')


special_tokens = {"bos_token" : "<bos>"
,"eos_token" : "<eos>"
,"unk_token" : "<unk>"
,"pad_token" : "<pad>"}

vocab_dict = {sp.id_to_piece(id): id for id in range(sp.get_piece_size())}

filtered_vocab = {token.replace('▁', ''): index for token, index in vocab_dict.items() if token != '<unk>'}

filtered_vocab["|"] = len(filtered_vocab)

for v in special_tokens.values():
            filtered_vocab[v] = len(filtered_vocab)
            
vocab = {token: index for index, (token, _) in enumerate(filtered_vocab.items())}

print(len(vocab))
print(vocab)


with open('pbe_vocab.json', 'w') as vocab_file:
    json.dump(vocab, vocab_file)