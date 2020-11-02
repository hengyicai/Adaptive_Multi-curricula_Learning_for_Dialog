#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains the main code for running CT and WD controlled models.
"""

import torch

from parlai.core.build_data import modelzoo_path
from projects.adaptive_learning.scripts.arora import SentenceEmbedder, load_arora
from projects.adaptive_learning.scripts.nidf import load_word2nidf
from projects.adaptive_learning.scripts.stopwords import STOPWORDS

# Interrogative words, used to control question-asking via weighted decoding
# From https://en.wikipedia.org/wiki/Interrogative_word
QN_WORDS = ['who', 'what', 'where', 'why', 'when', 'how', 'which', 'whom', 'whose', '?']


# ========================================
# LOADING NIDF MEASURES
# ========================================

class NIDFFeats(object):
    """
    An object to hold a vector containing the NIDF values for all words in the
    vocabulary. The vector is contstructed when first needed.
    """

    def __init__(self):
        self.NIDF_FEATS = None  # will be vector length vocab_size containing NIDF vals

    def make_feat_vec(self, dict):
        """
        Construct the NIDF feature vector for the given dict.
        """
        print("Constructing NIDF feature vector...")
        self.NIDF_FEATS = torch.zeros((len(dict)))
        num_oovs = 0
        for idx in range(len(dict)):
            word = dict[idx]
            if word in word2nidf:
                # Leave emoji (these appear in Twitter dataset) as NIDF=0
                # (so we don't encourage emoji when we set WD weight high for NIDF)
                if word[0] == '@' and word[-1] == '@':
                    continue
                nidf = word2nidf[word]  # between 0 and 1
                self.NIDF_FEATS[idx] = nidf
            else:
                # print("WARNING: word %s has no NIDF; marking it as NIDF=0" % word)
                num_oovs += 1  # If we don't have NIDF for this word, set as 0
        print('Done constructing NIDF feature vector; of %i words in dict there '
              'were %i words with unknown NIDF; they were marked as NIDF=0.'
              % (len(dict), num_oovs))

    def get_feat_vec(self, dict):
        """
        Return the NIDF feature vector. If necessary, construct it first.
        """
        if self.NIDF_FEATS is None:
            self.make_feat_vec(dict)
        return self.NIDF_FEATS


word2nidf = None
nidf_feats = None
arora_data = None
sent_embedder = None


def initialize_control_information(opt):
    """
    Loads information from word2count.pkl, arora.pkl in data/controllable_dialogue, and
    uses it to initialize objects for computing NIDF and response-relatedness controls.

    By default (build_task=True) we will also build the controllable_dialogue task i.e.
    download data/controllable_dialogue if necessary.
    """
    global word2nidf, nidf_feats, arora_data, sent_embedder

    if word2nidf is not None:
        # already loaded, no need to do anything
        return

    print("Loading up controllable features...")
    word2nidf = load_word2nidf(opt)  # get word2nidf dict
    nidf_feats = NIDFFeats()  # init the NIDFFeats object
    # load info for arora sentence embeddings
    arora_data = load_arora(opt)
    sent_embedder = SentenceEmbedder(
        arora_data['word2prob'],
        arora_data['arora_a'],
        arora_data['glove_name'],
        arora_data['glove_dim'],
        arora_data['first_sv'],
        glove_cache=modelzoo_path(opt['datapath'], 'models:glove_vectors'),
    )


# ========================================
# UTIL
# ========================================

def flatten(list_of_lists):
    """Flatten a list of lists"""
    return [item for sublist in list_of_lists for item in sublist]


def intrep_frac(lst):
    """Returns the fraction of items in the list that are repeated"""
    if len(lst) == 0:
        return 0
    num_rep = 0
    for idx in range(len(lst)):
        if lst[idx] in lst[:idx]:
            num_rep += 1
    return num_rep / len(lst)


def extrep_frac(lst1, lst2):
    """Returns the fraction of items in lst1 that are in lst2"""
    if len(lst1) == 0:
        return 0
    num_rep = len([x for x in lst1 if x in lst2])
    return num_rep / len(lst1)


def get_ngrams(text, n):
    """Returns all ngrams that are in the text.
    Inputs:
        text: string
        n: int
    Returns:
        list of strings (each is a ngram)
    """
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - (n - 1))]  # list of str


# ========================================
# SENTENCE-LEVEL ATTRIBUTE FUNCTIONS
# Given an input utterance, these functions compute the value of the controllable
# attribute at the sentence level (more precisely, at the utterance level).
#
# All these functions have the following inputs and outputs:
#
# Inputs:
#   utt: a string, tokenized and lowercase
#   history: list of strings. This represents the conversation history.
# Output:
#   score: float. the value of the controllable attribute for utt.
# ========================================

def intrep_repeated_word_frac(utt, history, remove_stopwords):
    """
    Sentence-level attribute function. See explanation above.
    Returns the fraction of words in utt that are repeated.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed before counting repetition.
    """
    assert utt.strip() != ""
    tokens = utt.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return intrep_frac(tokens)


def intrep_repeated_ngram_frac(utt, history, n):
    """
    Sentence-level attribute function. See explanation above.
    Returns the fraction of n-grams in utt that are repeated.
    Additional inputs:
      n: int, the size of the n-grams considered.
    """
    assert utt.strip() != ""
    ngrams = get_ngrams(utt, n)
    return intrep_frac(ngrams)


def extrep_repeated_word_frac(utt, history, remove_stopwords, person):
    """
    Sentence-level attribute function. See explanation above.
    Returns the fraction of words in utt that already appeared in a previous utterance.
    Additional inputs:
      remove_stopwords: bool. If True, stopwords are removed from utt before counting
        repetition.
      person: If 'self', identify words that have already been used by self (bot).
        If 'partner', identify words that have already been used by partner (human).
    """
    assert utt.strip() != ""
    if person == 'self':
        if len(history) >= 2:
            prev_utts = history[::-1][1::2][::-1]  # should already be tokenized
        else:
            prev_utts = []
    elif person == 'partner':
        prev_utts = history[::-1][0::2][::-1]  # should already be tokenized
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return 0
    tokens = utt.split()  # list of strings
    if remove_stopwords:  # remove stopwords from utt
        tokens = [t for t in tokens if t not in STOPWORDS]
    prev_words = [s.split() for s in prev_utts]  # list of list of ints
    prev_words = list(set(flatten(prev_words)))  # list of ints, no duplicates
    return extrep_frac(tokens, prev_words)


def extrep_repeated_ngram_frac(utt, history, n, person):
    """
    Sentence-level attribute function. See explanation above.
    Returns fraction of n-grams in utt that already appeared in a previous utterance.
    Additional inputs:
      n: int, the size of the n-grams considered.
      person: If 'self', identify n-grams that have already been used by self (bot).
        If 'partner', identify n-grams that have already been used by partner (human).
    """
    assert utt.strip() != ""
    if person == 'self':
        if len(history) >= 2:
            prev_utts = history[::-1][1::2][::-1]  # should already be tokenized
        else:
            prev_utts = []
    elif person == 'partner':
        prev_utts = history[::-1][0::2][::-1]  # should already be tokenized
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return 0
    utt_ngrams = get_ngrams(utt, n)
    prev_ngrams = [get_ngrams(prev, n) for prev in prev_utts]  # list of list of strings
    prev_ngrams = list(set(flatten(prev_ngrams)))  # list of strings, no duplicates
    return extrep_frac(utt_ngrams, prev_ngrams)


def avg_nidf(utt, history):
    """
    Sentence-level attribute function. See explanation above.
    Returns the mean NIDF of the words in utt.
    """
    words = utt.split()
    problem_words = [w for w in words if w not in word2nidf]
    ok_words = [w for w in words if w in word2nidf]
    if len(ok_words) == 0:
        print("WARNING: For all the words in the utterance '%s', we do not have the "
              "NIDF score. Marking as avg_nidf=1." % utt)
        return 1  # rarest possible sentence
    nidfs = [word2nidf[w] for w in ok_words]
    avg_nidf = sum(nidfs) / len(nidfs)
    if len(problem_words) > 0:
        print("WARNING: When calculating avg_nidf for the utterance '%s', we don't "
              "know NIDF for the following words: %s" % (utt, str(problem_words)))
    assert avg_nidf >= 0 and avg_nidf <= 1
    return avg_nidf


def contains_qmark(utt, history):
    """
    Sentence-level attribute function. See explanation above.
    Returns 1 if utt contains a question mark, otherwise 0.
    """
    return int("?" in utt)


def lastutt_sim_arora_sent(utt, history):
    """
    Sentence-level attribute function. See explanation above.

    Returns
      cos_sim(sent_emb(last_utt), sent_emb(utt))
    the cosine similarity of the Arora-style sentence embeddings for the current
    response (utt) and the partner's last utterance (last_utt, which is in history).

    - If there is no last_utt (i.e. utt is the first utterance of the conversation),
      returns None.
    - If one or both of utt and last_utt are all-OOV; thus we can't compute sentence
      embeddings, return the string 'oov'.
    """
    partner_utts = history[::-1][0::2][::-1]
    if len(partner_utts) == 0:
        # print('WARNING: returning lastuttsim = None because bot goes first')
        return None
    last_utt = partner_utts[-1]  # string
    if "__SILENCE__" in last_utt:
        assert last_utt.strip() == "__SILENCE__"
        # print('WARNING: returning lastuttsim = None because bot goes first')
        return None

    # Get sentence embeddings. Here we're naively splitting last_utt and utt; this is
    # fine given that we assume both utt and history are lowercase and tokenized.
    # Both last_utt_emb and response_emb are tensors length glove_dim (or None)
    last_utt_emb = sent_embedder.embed_sent(last_utt.split())
    response_emb = sent_embedder.embed_sent(utt.split())
    if last_utt_emb is None or response_emb is None:
        return 'oov'

    sim = torch.nn.functional.cosine_similarity(last_utt_emb, response_emb, dim=0)
    return sim.item()


def wordlist_frac(utt, history, word_list):
    """
    Sentence-level attribute function. See explanation above.
    Returns the fraction of words in utt that are in word_list.
    Additional inputs:
      word_list: list of strings.
    """
    words = utt.split()
    num_in_list = len([w for w in words if w in word_list])
    return num_in_list / len(words)


# In this dict, the keys are the names of the sentence-level attributes, and the values
# are functions with input (utt, history), returning the attribute value measured on utt
ATTR2SENTSCOREFN = {

    # Proportion of words in utt that appear earlier in utt
    "intrep_word":
        (lambda x: intrep_repeated_word_frac(x[0], x[1], remove_stopwords=False)),

    # Proportion of non-stopwords in utt that appear earlier in utt
    "intrep_nonstopword":
        (lambda x: intrep_repeated_word_frac(x[0], x[1], remove_stopwords=True)),

    # Proportion of 2-grams in utt that appear earlier in utt
    "intrep_2gram":
        (lambda x: intrep_repeated_ngram_frac(x[0], x[1], n=2)),

    # Proportion of 3-grams in utt that appear earlier in utt
    "intrep_3gram":
        (lambda x: intrep_repeated_ngram_frac(x[0], x[1], n=3)),

    # Proportion of words in utt that appeared in a previous bot utterance
    "extrep_word":
        (lambda x: extrep_repeated_word_frac(x[0], x[1], remove_stopwords=False,
                                             person='self')),

    # Proportion of non-stopwords in utt that appeared in a previous bot utterance
    "extrep_nonstopword":
        (lambda x: extrep_repeated_word_frac(x[0], x[1], remove_stopwords=True,
                                             person='self')),

    # Proportion of 2-grams in utt that appeared in a previous bot utterance
    "extrep_2gram":
        (lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=2, person='self')),

    # Proportion of 3-grams in utt that appeared in a previous bot utterance
    "extrep_3gram":
        (lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=3, person='self')),

    # Proportion of words in utt that appeared in a previous partner utterance
    "partnerrep_word":
        (lambda x: extrep_repeated_word_frac(x[0], x[1], remove_stopwords=False,
                                             person='partner')),

    # Proportion of non-stopwords in utt that appeared in a previous partner utterance
    "partnerrep_nonstopword":
        (lambda x: extrep_repeated_word_frac(x[0], x[1], remove_stopwords=True,
                                             person='partner')),

    # Proportion of 2-grams in utt that appeared in a previous partner utterance
    "partnerrep_2gram":
        (lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=2, person='partner')),

    # Proportion of 3-grams in utt that appeared in a previous partner utterance
    "partnerrep_3gram":
        (lambda x: extrep_repeated_ngram_frac(x[0], x[1], n=3, person='partner')),

    # Mean NIDF score of the words in utt
    "avg_nidf":
        (lambda x: avg_nidf(x[0], x[1])),

    # 1 if utt contains '?', 0 otherwise
    "question":
        (lambda x: contains_qmark(x[0], x[1])),

    # Proportion of words in utt that are interrogative words
    "qn_words":
        (lambda x: wordlist_frac(x[0], x[1], word_list=QN_WORDS)),

    # Cosine similarity of utt to partner's last utterance
    "lastuttsim":
        (lambda x: lastutt_sim_arora_sent(x[0], x[1])),
}


def eval_attr(utt, history, next_post, attr):
    """
    Given a conversational history and an utterance, compute the requested
    sentence-level attribute for utt.

    Inputs:
        utt: string. The utterance, tokenized and lowercase
        history: list of string. This represents the conversation history.
        attr: string. The name of the sentence-level attribute.
    Returns:
        value: float. The value of the attribute for utt.
    """
    # Check everything is lowercased already
    assert utt == utt.lower()
    for line in history:
        assert line == line.lower()
    if attr == 'post_sim':
        return ATTR2SENTSCOREFN['lastuttsim']((utt, [next_post]))
    # Eval attribute
    return ATTR2SENTSCOREFN[attr]((utt, history))


if __name__ == '__main__':
    pass
