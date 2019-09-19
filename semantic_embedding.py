import argparse
import sys
import pandas as pd
import pandas.errors
import os

cache_path = '.cache/cache.csv'


def cache_embeddings(words, model_path, **kwargs):
    print(words)
    cache = _load_embeddings(cache_path)
    cache['keys'] = cache[0].astype(str)
    embeddings = _load_embeddings(model_path)
    embeddings['keys'] = embeddings[0].str.lower().astype(str)
    targets = embeddings[embeddings['keys'].str.contains("|".join(words))].copy()
    print("cache", cache['keys'])
    print("targets", targets['keys'])
    if cache is None:
        cache = targets
    else:
        cache = cache.merge(targets, on='keys', how='right')
    print("cache", cache['keys'].head())
    _dump_embeddings(cache, cache_path)


def _load_embeddings(path):
    print("Loading embeddings from {}".format(path))
    try:
        try:
            df = pd.read_pickle(get_pickle_path(path))
        except (FileNotFoundError, EOFError):
            df = pd.read_csv(path, sep=" ", header=None, line_terminator="\n")
            df.to_pickle(get_pickle_path(path))
        print("Loaded embeddings from {}".format(path))
        return df
    except pandas.errors.EmptyDataError:
        return None


def _dump_embeddings(df, path):
    df.to_csv(path, sep=" ", header=False, index=False)


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
    if endpoint == "cache_embeddings":
        parser.add_argument(
            'words',
            nargs='+',
            type=str,
            help="Path to list of words to save embeddings for"
        )
        parser.add_argument(
            '--model_path',
            type=str,
            default='models/glove.840B.300d.txt',
            help="Path to GloVe mapping to use"
        )
        method = cache_embeddings
    elif endpoint == "get_similarity":
        parser.add_argument(
            'first',
            type=str,
            help="the first word"
        )
        parser.add_argument(
            'second',
            type=str,
            help="the second word"
        )
    else:
        raise ValueError("invalid endpoint")
    args = parser.parse_args()
    method(**vars(args))
