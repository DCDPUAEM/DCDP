from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def preprocesar_textos(docs,ignore_list:list=[]):
    strings_list = [re.sub(r'\n', ' ', x.lower()) for x in docs]
    strings_list = [re.sub(r'[^\w\s]', '', x) for x in strings_list]  # quita signos de puntuación
    strings_list = [re.sub('[0-9]', '', x) for x in strings_list] # quita números
    SW = stopwords.words('english') 
    for x in ignore_list: # quitamos stopwords
        SW.remove(x)
    tokens_no_sw = [" ".join([word for word in word_tokenize(text) if not word in SW]) for
                         text in strings_list ]
    return tokens_no_sw
