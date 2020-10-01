
import os
import sys
from pathlib import PosixPath
if sys.version_info < (3, 5):
    raise RuntimeError('DrQA supports Python 3.5 or higher.')

DATA_DIR = (
    os.path.join(PosixPath(__file__).absolute().parents[1].as_posix(), 'data')
)


from . import tokenizers
from . import retriever
from . import reader
from . import pipeline


# from .retriever import searchWikiArticle
# from .pipeline import create_drqa_instance