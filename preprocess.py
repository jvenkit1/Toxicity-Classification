import pandas as pd
from tqdm import tqdm

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def preprocess():
    """
    Preprocesses the dataset and generates word embeddings
    """
    train_data=pd.read_csv('data/train.csv')
    print(train_data.columns)
    print(train_data.shape)
    vocab = build_vocab(list(train_data['comment_text'].apply(lambda x:x.split())))
    print(len(vocab))

if __name__=="__main__":
    preprocess()
