import argparse
import sys
import pandas as pd
import pandas.errors
from sklearn.metrics.pairwise import cosine_similarity
import os

cache_path = '.cache/cache.csv'


def get_similarity(words, model_path, **kwargs):
    return cosine_similarity(_query_cache(words, model_path)[:, 1:])


def _query_cache(words, model_path):
    cache = _load_embeddings(cache_path)
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
            print("Warning: Multiple matches in cache for {}. Choose an expression with a unique match.".format(word))
    cache_embeddings(to_cache, model_path)
    return _regex_df_column(words, cache, 'keys').values


def cache_embeddings(words, model_path, **kwargs):
    """
    Finds and caches new embeddedings for easy access

    Example args: `cache_embeddings ^leader\W$ ^trustworthy$`

    :param words: a list of regex tokens to match semantic embeddings to be added to the cache
    :param model_path: the path of the GloVe model .txt file to use
    :param kwargs: unused args
    :return: the updated cache
    """
    print("Caching {}".format(words))
    if len(words) < 1:
        return
    cache = _load_embeddings(cache_path)
    embeddings = _load_embeddings(model_path, cache=True)
    targets = _regex_df_column(words, embeddings, 'keys').copy()
    if cache is None:
        cache = targets
    else:
        cache = cache.merge(targets, on='keys', how='right')
    _dump_embeddings(cache, cache_path)
    return cache


def _regex_df_column(expressions, df, column):
    if len(expressions) < 1:
        return None
    if len(expressions) == 1:
        return df[df[column].str.contains(expressions[0])]
    return df[df[column].str.contains("|".join(expressions))]


def _load_embeddings(path, cache=False):
    print("Loading embeddings from {}".format(path))
    try:
        if cache:
            try:
                df = pd.read_pickle(get_pickle_path(path))
                return df
            except (FileNotFoundError, EOFError):
                pass
        df = pd.read_csv(path, sep=" ", header=None)
        df.rename(columns={0: 'keys'}, inplace=True)
        df['keys'] = df['keys'].str.lower().astype(str)
        if cache:
            df.to_pickle(get_pickle_path(path))
        return df
    except pandas.errors.EmptyDataError:
        return None


def _dump_embeddings(df, path):
    df.to_csv(path, sep=" ", header=False, index=False)
    print("Dumped embeddings to {}".format(path))


def get_pickle_path(path):
    return '.cache/{}.pkl'.format(os.path.splitext(os.path.basename(path))[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the similarity of two words using semantic embeddings.")
    parser.add_argument(
        'endpoint',
        type=str,
        help="the endpoint to run"
    )
    endpoint = sys.argv[1]
    method = None

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/glove.840B.300d.txt',
        help="Path to GloVe mapping to use"
    )
    if endpoint in ["cache_embeddings", "get_similarity"]:
        parser.add_argument(
            'words',
            nargs='+',
            type=str,
            help="Path to list of words to save embeddings for"
        )
    if endpoint == "cache_embeddings":
        method = cache_embeddings
    elif endpoint == "get_similarity":
        method = get_similarity
    else:
        raise ValueError("invalid endpoint")
    args = parser.parse_args()
    print(method(**vars(args)))
