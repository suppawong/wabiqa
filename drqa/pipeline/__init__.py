# from ..reader.vector import batchify
# from ..reader.data import ReaderDataset, SortedBatchSampler
# from ..reader import DocReader
# from ..tokenizers import NewmmTokenizer
# from ..retriever import searchWikiArticle

import os
from ..tokenizers import NewmmTokenizer
from .. import DATA_DIR

DEFAULTS = {
    'tokenizer': NewmmTokenizer,
    'reader_model': os.path.join(DATA_DIR, 'model/drqa/'),
}


from .pipeline import create_drqa_instance