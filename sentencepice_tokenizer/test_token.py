from tokenizer import Wav2Vec2WordpieceTokenizer
from transformers import Wav2Vec2CTCTokenizer


special_tokens = {"bos_token" : "<bos>"
,"eos_token" : "<eos>"
,"unk_token" : "<unk>"
,"pad_token" : "<pad>"}
tokenizer = Wav2Vec2CTCTokenizer(vocab_file="./vocab.json", **special_tokens, word_delimiter_token="|")

text = """
$0a_0 zhe_4 shi_0 shi_0 yao_1 yi_0 si_0 ni_0 jiu_0 shi_0 yve_4 $0a_0 di_4 yi_0 ge_0 jiu_0 shi_0 ta_1 shou_4 dao_4 yi_4 qing_0 di_0 chong_4 ji_1 li_4 li_0 ying_3 xiang_3 geng_4 da_0 yin_1 wei_0 di_1 duan_1 fu_0 wu_0 ye_4 geng_4 duo_1 di_0 shi_0 yi_0 xie_1 xian_0 xia_0 xiao_0 fei_4 huo_0 dong_4 ta_1 bi_4 xv_1 de_0 mian_0 dui_0 ke_4 hu_0 di_0 bi_3 ru_2 yve_4 $0a_0 xiang_4 
di_4 yi_0 ge_0 jiu_0 shi_0 ta_1 shou_4 dao_4 yi_4 qing_0 di_0 chong_4 ji_1 li_4 li_0 ying_3 xiang_3 geng_4 da_0 yin_1 wei_0 di_1 duan_1 fu_0 wu_0 ye_4 geng_4 duo_1 di_0 shi_0 yi_0 xie_1 xian_0 xia_0 xiao_0 fei_4 huo_0 dong_4 ta_1 bi_4 xv_1 de_0 mian_0 dui_0 ke_4 hu_0 di_0 bi_3 ru_2 yve_4 $0a_0 xiang_4 $0e_0 di_0 fang_1 fu_0 wu_0 ye_4 di_0 yi_0 ge_0 jiu_0 ye_4 $0a_0 huan_2 you_3 huan_2 you_3 yi_0 xie_1 qi_2 ta_1 di_0 yi_0 xie_1 zhe_4 ge_0 
$0e_0 di_0 fang_1 fu_0 wu_0 ye_4 di_0 yi_0 ge_0 jiu_0 ye_4 $0a_0 huan_2 you_3 huan_2 you_3 yi_0 xie_1 qi_2 ta_1 di_0 yi_0 xie_1 zhe_4 ge_0 $0a_0 bi_3 ru_2 yve_4 zhe_4 ge_0 kai_1 you_1 bu_4 $0a_0 bi_3 ru_2 yve_4 zuo_4 yi_0 xie_1 zhe_4 zhong_3 
$0a_0 bi_3 ru_2 yve_4 zhe_4 ge_0 kai_1 you_1 bu_4 $0a_0 bi_3 ru_2 yve_4 zuo_4 yi_0 xie_1 zhe_4 zhong_3 zhe_4 xie_1 du_1 zhu_3 yv_2 $0a_0 xiang_0 dui_0 lai_2 jiang_3 shi_0 zhong_4 di_1 duan_1 fu_0 wu_0 ye_4 ne_4 yao_1 zhe_4 yi_0 bu_4 fen_0 jiu_0 ye_4 ni_0 ta_1 shi_0 geng_4 rong_2 yi_4 shou_4 dao_4 yi_4 qing_0 ying_3 xiang_3 di_0 yin_1 wei_0 jv_1 jie_0 yi_0 ge_2 li_2 ne_4 yao_1 xian_0 xia_0 xiao_0 fei_4 jiu_0 mo_4 you_3 liao_3 
zhe_4 xie_1 du_1 zhu_3 yv_2 $0a_0 xiang_0 dui_0 lai_2 jiang_3 shi_0 zhong_4 di_1 duan_1 fu_0 wu_0 ye_4 ne_4 yao_1 zhe_4 yi_0 bu_4 fen_0 jiu_0 ye_4 ni_0 ta_1 shi_0 geng_4 rong_2 yi_4 shou_4 dao_4 yi_4 qing_0 ying_3 xiang_3 di_0 yin_1 wei_0 jv_1 jie_0 yi_0 ge_2 li_2 ne_4 yao_1 xian_0 xia_0 xiao_0 fei_4 jiu_0 mo_4 you_3 liao_3 $0a_0 di_4 $0er_4 ge_0 jiu_0 shi_0 mei_3 guo_2 di_0 ta_1 di_0 zhe_4 ge_0 jv_1 min_2 bu_4 men_2 ni_0 $0e_0 huan_2 you_3 yi_0 ge_0 huan_2 huan_2 you_3 yi_0 ge_0 wen_4 ti_2 jiu_0 shi_0 $0a_0 ta_1 di_0 jie_0 gou_4 ta_1 di_0 jie_0 gou_4 te_4 zheng_1 fei_1 chang_2 di_0 zhe_4 ge_0 fen_0 hua_4 $0a_0 
$0a_0 di_4 $0er_4 ge_0 jiu_0 shi_0 mei_3 guo_2 di_0 ta_1 di_0 zhe_4 ge_0 jv_1 min_2 bu_4 men_2 ni_0 $0e_0 huan_2 you_3 yi_0 ge_0 huan_2 huan_2 you_3 yi_0 ge_0 wen_4 ti_2 jiu_0 shi_0 $0a_0 ta_1 di_0 jie_0 gou_4 ta_1 di_0 jie_0 gou_4 te_4 zheng_1 fei_1 chang_2 di_0 zhe_4 ge_0 fen_0 hua_4 $0a_0 jiu_0 shi_0 fu_0 wu_0 ye_4 li_0 mian_0 $0a_0 xin_1 zeng_1 liao_3 fei_1 nong_2 fu_0 wu_0 ye_4 li_0 mian_0 you_3 yi_0 bu_4 fen_0 shi_0 jin_1 rong_2 di_0 ke_1 ji_4 ne_4 yao_1 jin_1 rong_2 ke_1 ji_4 da_0 jie_0 du_1 zhi_4 dao_0 zhe_4 yi_0 kuai_4 shou_4 dao_4 yi_4 qing_0 di_0 ying_3 xiang_3 shi_0 hen_3 xiao_3 di_0 yin_1 wei_0 du_1 shi_0 yi_0 xie_1 shang_0 ban_1 gong_1 dan_4 shi_0 
jiu_0 shi_0 fu_0 wu_0 ye_4 li_0 mian_0 $0a_0 xin_1 zeng_1 liao_3 fei_1
"""

output = tokenizer.tokenize(text)

print(output)
print(len(output))