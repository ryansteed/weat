import sys
import argparse
import pandas as pd
from util.embeddings import SemanticEmbedding, EmbeddingCache


def get_similarity_matrix(words, model_path, **kwargs):
    return SemanticEmbedding.get_similarity_matrix(words, model_path)


def cache_embeddings(words, model_path, **kwargs):
    """
    Example args: `cache_embeddings ^leader\W$ ^trustworthy$`
    """
    return EmbeddingCache.cache_embeddings(words, model_path)


def imagenet_similarity_matrix(parent, id_word_path, parent_child_path, model_path, **kwargs):
    id_word = file_to_dict(id_word_path, sep="\t")
    print(id_word)
    parent_child = pd.read_csv(parent_child_path, sep="\t", header=None)
    print(parent_child)
    children = parent_child[parent_child[0] == parent]
    print(children)


def file_to_dict(path, sep=" "):
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.strip("\n").split(sep)
            d[key] = val
    return d


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
    print(endpoint)
    if endpoint == "imagenet_similarity_matrix":
        parser.add_argument(
            'parent',
            type=str,
            help="Synset ID of parent to gather child IDs for"
        )
        parser.add_argument(
            'id_word_path',
            type=str,
            help="Path to Synset ID / word table"
        )
        parser.add_argument(
            'parent_child_path',
            type=str,
            help="Path to parent ID / child ID table"
        )
        method = imagenet_similarity_matrix
    elif endpoint == "cache_embeddings":
        method = cache_embeddings
    elif endpoint == "get_similarity":
        method = get_similarity_matrix
    else:
        raise ValueError("invalid endpoint")
    args = parser.parse_args()
    print(method(**vars(args)))
