import pandas as pd
import numpy as np
import unicodedata
import string
import re
import nltk

from gensim.models import FastText
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import umap

class txt2clust(object):

    def __init__(self,
                 size_embeddor=40,
                 size_dim_reduction = 5,
                 num_cluster="auto",
                 txt_embeddor="fasttext",
                 only_embedding = False,
                 create_clusters_info = False,
                 text_preprocessing = "other",
                 stopwords = [],
                 *args,
                 **kwargs):
        

        self._is_trained = False
        self.clusters_info = {}
        self.scaler = MinMaxScaler()
        self.stopwords = self.check_stopwords(stopwords)
        self.text_preprocessing = self.check_text_preprocessing(text_preprocessing)
        self.only_embedding = self.check_only_embedding(only_embedding)
        self.num_cluster = self.check_num_cluster(num_cluster)
        self.gm = GaussianMixture(n_components=self.num_cluster, random_state=2020)
        self.dbscan = DBSCAN(n_jobs=-1)
        self.size_embeddor = self.check_size_embeddor(size_embeddor)
        self.txt_embeddor = self.check_txt_embeddor(txt_embeddor)
        self.size_dim_reduction = self.check_size_dim_reduction(size_dim_reduction)
        self.create_clusters_info = create_clusters_info
        self.reductionizer = umap.UMAP(n_components=self.size_dim_reduction, random_state=2020, n_neighbors=30, min_dist=0.0)

                

    def check_data(self, data):
        if not isinstance(data, pd.Series):
            raise Exception('input data must be a pandas.Series')
                
    def check_stopwords(self, stopwords):
        if type(stopwords) == list:
            for stopword in stopwords:
                if type(stopword) == str:
                        pass
                else:
                    raise Exception('stopwords must be a list of string')
        else:
            raise Exception('stopwords must be a list of string')
        return stopwords

                

    def check_only_embedding(self, only_embedding):
        if (only_embedding == False) or (only_embedding == True):
            pass
        else:
            raise Exception('only_embedding must be set at True or False')
        return only_embedding
    
                
    def check_num_cluster(self, num_cluster):
        if (num_cluster == "auto") or ((type(num_cluster) == int) and (num_cluster > 0)):
            return num_cluster
        else:
            raise Exception('num_cluster must be "auto" or positive integer')
            
            
    def check_size_embeddor(self, size_embeddor):
        if type(size_embeddor) == int:
            if size_embeddor > 0:
                return size_embeddor
            else:
                raise Exception('size_embeddor "{}" must be > 0'.format(size_embeddor))
        else:
            raise Exception('size_embeddor "{}" must be int'.format(size_embeddor))
            
            
    def check_size_dim_reduction(self, size_dim_reduction):
        if type(size_dim_reduction) == int:
            if size_dim_reduction > 0:
                return size_dim_reduction
            else:
                raise Exception('size_dim_reduction "{}" must be > 0'.format(size_dim_reduction))
        else:
            raise Exception('size_dim_reduction "{}" must be int'.format(size_dim_reduction))
            
            
    def check_txt_embeddor(self, txt_embeddor):
        if txt_embeddor == "fasttext":
            txt_embeddor = FastText(size=self.size_embeddor, window=5,min_count = 1,sample=1e-3, sg=1, workers=1)

        if isinstance(txt_embeddor, FastText):
            return txt_embeddor
        else:
            raise Exception('Wrong txt_embeddor parameter : must be "fasttext" or instances of gensim.models.FastText')

            
    def check_text_preprocessing(self, text_preprocessing):
        
        if (text_preprocessing == "latin"):
            self.accented_char_removal=True
            self.text_lower_case=True
            self.special_char_removal=True
            self.remove_digits=True
            self.stemmer_contraction = True
            
        if (text_preprocessing == "other"):
            self.accented_char_removal=False
            self.text_lower_case=True
            self.special_char_removal=True
            self.remove_digits=True
            self.stemmer_contraction = False
            
        if (text_preprocessing == "raw"):
            self.accented_char_removal=False
            self.text_lower_case=False,
            self.special_char_removal=False, 
            self.remove_digits=False  
            self.stemmer_contraction = False

        if (text_preprocessing not in ["latin","other","raw"]):
            raise Exception('text_preprocessing must be set at "latin" or "other" or "raw"')
            
        return text_preprocessing
    
                
    def fit(self, serie):
        self.check_data(serie)
        data = pd.DataFrame()
        data["base_txt"] = serie.astype(str)
        data["normalized_txt"] = self.normalize_corpus(serie)
        data = pd.concat([data, self.get_embedding(data["normalized_txt"])], axis=1)
        
        if self.only_embedding == False:
            data = self.get_reduction_features(data.copy())
            
            if self.create_clusters_info == True:
                self.create_clusters_infos(data.copy())
            
        self._is_trained = True
        
    
    def predict(self,serie):
        
        if self._is_trained != True:
            raise Exception('txt2clust is not trained')
            
        self.check_data(serie)
        data = pd.DataFrame()
        data["base_txt"] = serie.astype(str)
        data["normalized_txt"] = self.normalize_corpus(serie)
        data = pd.concat([data, self.get_embedding(data["normalized_txt"])], axis=1)
        
        if self.only_embedding == False:
            data = self.get_reduction_features(data.copy())
            
        return data

    def predict_proba(self,data):
        data = self.predict(data)
        cols = [col for col in data.columns if "proba_cluster" in col]
        
        return data[cols]
    
    def predict_cluster(self,data):
        data = self.predict(data)
        cols = [col for col in data.columns if "cluster_labels" in col]
        
        return data[cols]
    
    def predict_embedding(self,data):
        data = self.predict(data)
        cols = [col for col in data.columns if "embedding_" in col]
        
        return data[cols]
    
    def predict_reduct_embedding(self,data):
        data = self.predict(data)
        cols = [col for col in data.columns if "reduction_feature" in col]
        
        return data[cols]
    
    
    def create_clusters_infos(self, data):
        for i in data["cluster_labels"].unique():
            try:
                vectorizer = CountVectorizer(ngram_range=(1,2))
                Y = vectorizer.fit_transform(data[data["cluster_labels"] == i]["normalized_txt"])
                Y = pd.DataFrame(Y.toarray(), columns=vectorizer.get_feature_names())
                Y = pd.DataFrame(Y.sum().transpose().sort_values(ascending=False))
                self.clusters_info[i] = Y.head(20).index.to_list()
            except: pass

                
    def remove_accents(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text


    def remove_special_characters(self, text):
        pattern = r'[{}]'.format(string.punctuation) if not self.remove_digits else r'[{}0-9]'.format(string.punctuation)
        text = re.sub(pattern, ' ', text)
        return text


    def stemmer(self, text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    
    def remove_stopwords(self, text):
        tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        
        if self.text_lower_case:
            filtered_tokens = [token for token in tokens if token not in self.stopwords]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        filtered_text = ' '.join(filtered_tokens)    
        
        return filtered_text


    def normalize_corpus(self, corpus):

        normalized_corpus = []

        for doc in corpus:
            doc = str(doc)

            if self.accented_char_removal == True:
                doc = self.remove_accents(doc)

            if self.text_lower_case == True:
                doc = doc.lower()

            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

            if self.special_char_removal == True:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(doc)  

            doc = re.sub(' +', ' ', doc)
            doc = self.remove_stopwords(doc)

            if self.stemmer_contraction == True:            
                doc = self.stemmer(doc)

            doc =  re.sub(r"\b[a-zA-Z]\b", "", doc)
            normalized_corpus.append(doc)

        return pd.Series(normalized_corpus)

    
    def get_mean_sentence_embedding(self, tokenized_sentence, vocabulary):

        feature_vector = np.zeros((self.size_embeddor,), dtype="float32")
        tot_words = 0.

        for word in tokenized_sentence:
            if word in vocabulary:
                tot_words += 1.
                feature_vector = np.add(feature_vector, self.txt_embeddor[word])
                
        if tot_words:
            feature_vector = np.divide(feature_vector, tot_words)

        return feature_vector


    def get_sentence_embedding(self, corpus):

        vocabulary = set(self.txt_embeddor.wv.index2word)
        features = [self.get_mean_sentence_embedding(tokenized_sentence, vocabulary)
                        for tokenized_sentence in corpus]
        
        return np.array(features)

    def get_embedding(self, serie):

        wpt = nltk.WordPunctTokenizer()
        tokenized_corpus = [wpt.tokenize(document) for document in serie]

        if self._is_trained == False:
            self.txt_embeddor.build_vocab(sentences=tokenized_corpus)
            self.txt_embeddor.train(sentences=tokenized_corpus,
                                    total_examples=len(tokenized_corpus),
                                    epochs=100)

        vectors = self.get_sentence_embedding(corpus=tokenized_corpus)
        vectors = pd.DataFrame(vectors)
        
        for column in vectors.columns:
            vectors.rename(columns={column: str("embedding_")+str(column)}, inplace=True)

        return vectors.astype("float32")

    
    def get_reduction_features(self, data):
        cols = [col for col in data.columns if "embedding" in col]
        
        if self._is_trained == False:
            self.scaler.fit(data[cols].copy())
            scaled_data = self.scaler.transform(data[cols].copy())
            self.reductionizer.fit(scaled_data.copy())
            reduce_data = self.reductionizer.transform(scaled_data.copy())
            
            if self.num_cluster == "auto":                
                self.dbscan.fit(reduce_data.copy())
                self.num_cluster = np.max(self.dbscan.labels_) + 1
                
                if self.num_cluster == 0:
                    self.num_cluster +=  1
            self.gm = GaussianMixture(n_components=self.num_cluster, random_state=2020)
            self.gm.fit(reduce_data.copy())
            
        df= pd.DataFrame()
        data_pipe = self.scaler.transform(data[cols].copy())
        data_pipe = self.reductionizer.transform(data_pipe)
        reduction_features = pd.DataFrame(data_pipe, columns=["reduction_feature_"+str(i) for i in range(0, data_pipe.shape[1])])
        
        proba = self.gm.predict_proba(data_pipe)
        proba = pd.DataFrame(proba.astype("float16"), columns=["proba_cluster_"+str(i) for i in range(0, proba.shape[1])])
        df = pd.concat([reduction_features, proba], axis=1)
        
        df["cluster_labels"] =self.gm.predict(data_pipe).astype("int16")

        data = pd.concat([data,df], axis=1)

        return data
    
