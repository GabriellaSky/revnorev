from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings, FlairEmbeddings
from flair.data import Corpus, Sentence
import numpy as np
import pandas as pd
import tqdm


def initialize_emb(model):
    if model == 'glove':
        # initialize the claim embeddings, mode = mean
        glove_embedding = WordEmbeddings('glove')
        return DocumentPoolEmbeddings([glove_embedding])
    elif model == 'flair':
        flair_embedding_forward = FlairEmbeddings('news-forward')
        return DocumentPoolEmbeddings([flair_embedding_forward])
    else:
        # initialize the claim embeddings for any registered transformer model
        return TransformerDocumentEmbeddings(model)


def get_embeddings(claims, emb_model, prefix, model_name):
    emb = []
    for text in tqdm.tqdm(claims):
        sentence = Sentence(text)

        try:
            emb_model.embed(sentence)
            emb.append(sentence.embedding.cpu().numpy())
        except:
            if model_name == 'glove':
                emb.append(np.zeros(100))
            elif model_name == 'flair':
                emb.append(np.zeros(2048))
            else:
                emb.append(np.zeros(768))

    emb = pd.DataFrame(np.vstack(emb))
    emb = emb.add_prefix(prefix + "_")

    return emb

def get_multiclass_label(x):
    if x =='Typo or grammar correction':
        return 1
    elif x=='Clarified claim':
        return 2
    elif x =='Corrected or added links':
        return 0
    elif x == 'Clarified argument':
        return 2
    else:
        return 3

