"""
Modified version newmm tokenizer from PyThaiNlP

chakri.lowphansirikul@gmail.com
"""
import re
import logging
from pythainlp.tag import pos_tag
from pythainlp.ner import thainer
from pythainlp.tokenize import word_tokenize,dict_word_tokenize,create_custom_dict_trie
from .tokenizer import Tokens, Tokenizer
import os
import copy
from . import DEFAULTS

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
data_dict=create_custom_dict_trie(DEFAULTS['word_corpus_th'])

logger = logging.getLogger(__name__)

class NewmmTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
            substitutions: if true, normalizes some token types (e.g. quotes).
        """

        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))

        self.annotators = set(['pos','ner'])
     

        self.substitutions = kwargs.get('substitutions', True)

    

    def tokenize(self, text):
        def tokenize_in_list(tok_pos_ner, delimeter):
            new_tokens = []
            new_pos = []
            new_ner = []
            for item in tok_pos_ner:
                tokens_list = re.split(delimeter, item[0])

                tokens_list = list(filter(None, tokens_list))
                n = len(tokens_list)

                new_tokens.extend(tokens_list)

                new_pos.extend([item[1]] * n)
                
                # set only first token to B, second and so on set to I-
                ner_list = [item[2]] * n
                ner_list_second_to_end = ner_list[1:]
                ner_list_second_to_end = list(map(lambda x: str.replace(x, "B-", "I-"), ner_list_second_to_end))
                new_ner.extend([ner_list[0]] + ner_list_second_to_end)

            return new_tokens, new_pos, new_ner

        def charOffset(i, tokens):
            count = 0
            for idx, token in enumerate(tokens):
                for char in token:
                    if (idx == i):
                        start_ws = count
                        end_ws = count + len(tokens[i])
                        return start_ws, end_ws
                    count += 1
                    
        def cleanText(text):
            tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
            return tag_re.sub('', text)
        
        ner = thainer()

        data = []
        
        clean_text = cleanText(text)
        tokens =  ner.get_ner(clean_text)
        

        delimeters_list=['"','\)','\(','<','>','\s','”','“',';','-','ผู้ว่าราชการ',
                    'ใน','ของ','ศาลากลาง','ชาว','ถือ','ประเทศ',
                   'เจ้าผู้ครอง','กลาง','ดำรง','มหาวิทยาลัย','โรงเรียน',
                   'สี','วัน','โรค','การ','งาน','ความ','ภาษา','ศาสนา','ทวีป',
                    'ภาค','เพลง','ดอนัลด์','เป็น','ดอนัลด์','มือ','เพศ','ทิศ','ที่',
                    'ท่าน','รูป','หัวหน้า','กระทรวง','ริม','อักษร','ทิศ','เนื้อ',
                    'ผิว']
    
        tokens, tags, ents = tokenize_in_list(tokens, r'(' + '|'.join(delimeters_list) + ')')

 

        for i in range(len(tokens)):
            # Get text
            token = tokens[i]

            # Format data
            data.append((
                token,
                token,
                charOffset(i, tokens),
                tags[i], # pos tah
                token, # lemma
                ents[i] # entity
            ))
        return Tokens(data, self.annotators)
