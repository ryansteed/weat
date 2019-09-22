import pandas as pd
import pandas.errors
from sklearn.metrics.pairwise import cosine_similarity
import os

cache_path = '.cache/cache.csv'


class SemanticEmbedding:
    def __init__(self, token, model_path):
        self.token = token
        self.model_path = model_path
        self.embedding = self.get_embedding()[0, :]

    def get_embedding(self):
        return EmbeddingCache.query_cache([self.token], self.model_path)[:, 1:]

    @staticmethod
    def get_similarity_matrix(words, model_path):
        return cosine_similarity([SemanticEmbedding(word, model_path).embedding for word in words])


class EmbeddingCache:
    @staticmethod
    def cache_embeddings(words, model_path):
        """
        Finds and caches new embeddedings for easy access

        Example args: `cache_embeddings ^leader\W$ ^trustworthy$`

        :param words: a list of regex tokens to match semantic embeddings to be added to the cache
        :param model_path: the path of the GloVe model .txt file to use
        :return: the updated cache
        """
        print("Caching {}".format(words))
        if len(words) < 1:
            return
        cache = EmbeddingCache.load_embeddings(cache_path)
        embeddings = EmbeddingCache.load_embeddings(model_path, cache=True)
        targets = EmbeddingCache.regex_df_column(words, embeddings, 'keys').copy()
        if cache is None:
            cache = targets
        else:
            cache = cache.merge(targets, on='keys', how='right')
        EmbeddingCache.dump_embeddings(cache, cache_path)
        return cache

    @staticmethod
    def regex_df_column(expressions, df, column):
        if len(expressions) < 1:
            return None
        if len(expressions) == 1:
            return df[df[column].str.contains(expressions[0])]
        return df[df[column].str.contains("|".join(expressions))]

    @staticmethod
    def load_embeddings(path, cache=False):
        print("Loading embeddings from {}".format(path))
        try:
            if cache:
                try:
                    df = pd.read_pickle(EmbeddingCache._get_pickle_path(path))
                    return df
                except (FileNotFoundError, EOFError):
                    pass
            df = pd.read_csv(path, sep=" ", header=None)
            df.rename(columns={0: 'keys'}, inplace=True)
            df['keys'] = df['keys'].str.lower().astype(str)
            if cache:
                df.to_pickle(EmbeddingCache._get_pickle_path(path))
            return df
        except pandas.errors.EmptyDataError:
            return None

    @staticmethod
    def dump_embeddings(df, path):
        df.to_csv(path, sep=" ", header=False, index=False)
        print("Dumped embeddings to {}".format(path))

    @staticmethod
    def query_cache(words, model_path):
        cache = EmbeddingCache.load_embeddings(cache_path)
        if cache is None:
            print("No cache found. Generating new cache...")
            cache = cache_embeddings(words, model_path)
        to_cache = []
        for word in words:
            matches = cache[cache['keys'].str.contains(word)]
            print(word, matches.shape)
            if matches.shape[0] < 1:
                to_cache.append(word)
            if matches.shape[0] > 1:
                print(
                    "Warning: Multiple matches in cache for {}. Choose an expression with a unique match.".format(word))
        cache_embeddings(to_cache, model_path)
        return EmbeddingCache.regex_df_column(words, cache, 'keys').values

    @staticmethod
    def _get_pickle_path(path):
        return '.cache/{}.pkl'.format(os.path.splitext(os.path.basename(path))[0])
