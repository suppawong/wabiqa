#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from .. import DATA_DIR

DEFAULTS = {
    'word_corpus_th': os.path.join(DATA_DIR, 'corpus/words_th.txt'),
}

from .newmm_tokenizer import NewmmTokenizer


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

def get_class(name):
    if name == "newmm":
        return NewmmTokenizer
    
    # raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)
