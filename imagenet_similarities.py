import sys
import argparse
import pandas as pd
from util.embeddings import SemanticEmbedding, EmbeddingCache


def get_similarity_matrix(words, model_path, **kwargs):
    if len(words) == 2:
        return SemanticEmbedding.get_similarity_matrix(words, model_path)[0, 0]
    return SemanticEmbedding.get_similarity_matrix(words, model_path)


def cache_embeddings(words, model_path, **kwargs):
    """
    Example args: `cache_embeddings ^leader\W$ ^trustworthy$`
    """
    return EmbeddingCache.cache_embeddings(words, model_path)


def imagenet_similarity_matrix(parent, id_word_path, parent_child_path, model_path, **kwargs):
    id_word = pd.read_csv(id_word_path, sep="\t", header=None)
    id_word.columns = ["id", "token"]
    parent_child = pd.read_csv(parent_child_path, sep=" ", header=None)
    parent_child.columns = ["parent", "child"]
    print("Finding children for parent {}".format(parent))
    tokens = pd.merge(
        id_word, parent_child[parent_child["parent"] == parent],
        how="right", left_on="id", right_on="child"
    )[["id", "token"]]
    print(tokens)
    tokens["similarity_trustworthy"] = tokens["token"].apply(
        lambda x: get_similarity_matrix(["^{}$".format(x), "^trustworthy$"], model_path)
    )
    print(tokens)


# def file_to_dict(path, sep=" "):
#     d = {}
#     with open(path) as f:
#         for line in f:
#             (key, val) = line.strip("\n").split(sep)
#             d[key] = val
#     return d


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
