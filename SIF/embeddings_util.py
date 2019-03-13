
import os
from gensim.models import KeyedVectors
import numpy as np

def load_embeddings(we_type,path_file):
    wordvectors_vec_file = ''
    if we_type=='fasttext':
        print('Loading Fasttext embeddings...')
        wordvectors_vec_file = os.path.join(path_file)
    # if we_type == 'glove': 
    #     print('Loading Glove embeddings...')
    #     wordvectors_vec_file = EMBEDDING_FILE_GLOVE_VEC
    wordvectors = KeyedVectors.load_word2vec_format(wordvectors_vec_file)
    return wordvectors

def create_emb_dict(words, emb_dimension,wordvectors):
    emb_dict = {}
    size = len(words)
    for i in range(size):
        word = words[i]
        vector = np.array(wordvectors[words[i]], dtype=np.float32)
        if vector.shape[0]== emb_dimension:
            emb_dict[word] = vector
    return emb_dict
# array: ['de','la','a'...] of WordEmbeddings
def create_words_vocab(vocab):
    words = []
    for word in vocab:
        words.append(word)
    return words

# def create_word_idx_dict(words, emb_dimension):
#     emb_dict = {}
#     size = len(words)
#     for i in range(size):
#         word = words[i]
#         emb_dict[word]
# rs[words[i]], dtype=np.float32)
#         if vector.shape[0]== emb_dimension:
#             emb_dict[word] = vector
#     return emb_dict

### Load Word embeddings

############ Matching dataset vocab with embeddings vocab
#LOAD PRETRAINED VECTORS
#wordvectors = load_embeddings('fasttext','data/fasttext-sbwc.vec')
# emb_dict_fasttext = pickle.load(open("data/emb_dict_fasttext.pickle",'rb'))

# len(emb_dict_fasttext), emb_dict_fasttext

# sentences_list=[]
# #df_train.tweet = pefix_hash_tag(df_train.tweet)
# #df_train.tweet = add_prefix_lol(df_train.tweet)

# for sentence in df_train.tweet:
#     clean_text = remove_mentions(sentence)
#     clean_text = remove_urls(clean_text)
#     clean_text = remove_punctuation(clean_text)
#     #clean_text = remove_digits(clean_text)
#     clean_text = to_lower(clean_text)
# #     clean_text = remove_stopwords(clean_text)
#     #clean_text = processJaja(clean_text)
#     clean_text = remove_emoji(clean_text)
        
#     sentence_tokenize = word_tokenize(clean_text)
#     sentences_list.append(sentence_tokenize)
# sentences_list

# len(sentences_list),sentences_list
# tweets = np.array([flattern(tweet) for tweet in sentences_list])
# tweets = np.array([' '.join(i) for i in sentences_list])


# tweets

# from keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(tweets) 


# dataset_words = tokenizer.word_index
# dataset_words

# We_esp=[]
# words_dataset_esp = {}
# index = 0
# for word, _ in dataset_words.items():
#     vector = emb_dict_fasttext.get(word)
#     if vector is not None:
#         We_esp.append(vector)
#         words_dataset_esp[word] = index
#         index +=1
# We_esp = np.array(We_esp)
        
# We_esp.shape, len(words_dataset_esp)

# words_dataset_esp

# pickle.dump(words_dataset_esp, open('data/words_dataset_esp.pkl', 'wb'))
# np.save('data/We_esp.npy',We_esp)       