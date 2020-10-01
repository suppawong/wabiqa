import torch
import regex
import heapq
import math
import time
import logging
import sys
import re
import json
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from collections import Counter
from ..reader.vector import batchify
from ..reader.data import ReaderDataset, SortedBatchSampler
from ..reader.model import DocReader
from .. import reader
from .. import tokenizers
from .. import retriever
from . import DEFAULTS
from pythainlp.ner import thainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ner = thainer()
PROCESS_TOK = None
PROCESS_CANDS = None

def init(tokenizer_class, tokenizer_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_CANDS = candidates


# FOR TESTING
def load_mock_item(path):
    with open(path, encoding='utf-8') as file:
        data = json.load(file) 
    return data['article_id'], data['title'], data['context'], data['context_cleaned'], data['snippet'], data['pad']

def getContextFromQuestion(question, search_space=1, image=False):
    # For testing
    if question == 'ชวน หลีกภัย อดีตนายกรัฐมนตรีคนที่ 20 ของไทย เกิดเมื่อวันที่เท่าไร':
        return load_mock_item('../../tests/mock_item_1.json')

    if question == 'ยิ่งลักษณ์ ชินวัตร อดีตนายกรัฐมนตรีคนที่ 28 ของไทย เกิดวันที่เท่าไร':
        return load_mock_item('../../tests/mock_item_2.json')

    wikipedia_article_id, title, context, context_cleaned, snippet, pad = retriever.searchWikiArticle(question, search_space)
    return wikipedia_article_id, title, context, context_cleaned, snippet, pad



def tokenize_text(text):
    global PROCESS_TOK

    return PROCESS_TOK.tokenize(text)


def create_drqa_instance(model):
    pipeline = DrQA(
        cuda=False,
        fixed_candidates=None,
        reader_model=DEFAULTS['reader_model'] + model + '.mdl',
        # ranker_config={'options': {'tfidf_path': None}},
        # db_config={'options': {'db_path': None}},
        tokenizer='newmm'
    )
    return pipeline

class DrQA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            db_config=None,
            ranker_config=None
    ):
        """Initialize the pipeline.

        Args:
            reader_model: model file from which to load the DocReader.
            embedding_file: if given, will expand DocReader dictionary to use
              all available pretrained embeddings.
            tokenizer: string option to specify tokenizer used on docs.
            fixed_candidates: if given, all predictions will be constrated to
              the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use the gpu.
            data_parallel: whether to use multile gpus.
            max_loaders: max number of async data loading workers when reading.
              (default is fine).
            num_workers: number of parallel CPU processes to use for tokenizing
              and post processing resuls.
            db_config: config for doc db.
            ranker_config: config for ranker.
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        # logger.info('Initializing document ranker...')
        
        logger.info('Initializing document reader...')
        reader_model = reader_model
        self.reader = DocReader.load(reader_model, normalize=True)
        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.reader.expand_dictionary(words)
            self.load_embeddings(added, embedding_file)
        if cuda:
            self.cuda()
        if data_parallel:
            self.parallelize()

        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)
        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        # db_config = db_config or {}
        # db_class = db_config.get('class', DEFAULTS['db'])
        # db_opts = db_config.get('options', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, fixed_candidates)
        )

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def _split_doc_slicing_window(self, doc, size, stride):
        '''
            Params:
                doc: is string of document
                size:is number of tokens per windows
                stride: is number of tokens as stride
            Return:
                list of splited document based on window size and stride
        '''
        tokens = re.split(r'(\s+)', doc)
        n_tokens = len(tokens)
        splits = []
        begin = 0
        end = size
        i=0
        while(i < n_tokens):
            end = i+size
            if (end > n_tokens):
                end = n_tokens
            splits.append(''.join(tokens[begin:end]))
            begin = end - stride
            i += size
        return splits
        
    def _get_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )
        return loader

  
    def process_single(self, query, question_id, candidates=None, top_n=3, n_docs=1, reutrn_context=False):
        """Run a single query"""
        '''
            Params:
                question_id: id of the query can be None
                query: question in Thai natural language

            Return:
                question_id: as param if it is not None
                question: as param
                answer: answer to the question
                start_position: start position based on UNCLEANED Wiki article
                end_position: end position based on UNCLEANED Wiki article
                context: context of UNCLEAN Wiki article
        '''
        predictions = self.process_batch(
            [{'question': query, 'question_id': str(question_id)}], None, top_n, n_docs, False
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=3, n_docs=1, reutrn_context=False):
        """Run a single query"""
        '''
            Params:
                queries: is a list of dict {question_id: <str>, question: <str>}

            Return:
                question_id: as param queries
                question: as param queries
                answer: answer to the question
                start_position: start position based on UNCLEANED Wiki article
                end_position: end position based on UNCLEANED Wiki article
                context: context of UNCLEAN Wiki article
        '''
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)



        # 1. Retrieve the context of the wikipedia article in cleaned format (exclude <document> tag)
        list_query = []
        list_article_id = []
        list_context_cleaned = []
        list_context_uncleaned = []
        list_snippet = []
        list_pad = []
        list_query_question = []
        list_title = []
        # list_images_url = []

        for query in queries: 
            article_id, title, context, context_cleaned, snippet, pad = getContextFromQuestion(query['question'], image=False)
            if (article_id == 'not found'):
                print('article not found')
                list_query.append(query)
                list_query_question.append(query['question'])
                list_title.append(None)
                list_context_cleaned.append('')
                list_article_id.append(None)
                list_context_uncleaned.append('')
                list_snippet.append('')
                list_pad.append(0)
                # list_images_url.append(None)
                continue

            list_query.append(query)
            list_query_question.append(query['question'])
            list_title.append(title)
            list_context_cleaned.append(context_cleaned)
            list_article_id.append(article_id)
            list_context_uncleaned.append(context)
            list_snippet.append(snippet)
            list_pad.append(pad)
            # list_images_url.append(images_url)

        
        print('[DONE Fetching Document]')
        # did2didx = {did: didx for didx, did in enumerate(list_context_cleaned)}
        did2didx = {did: didx for didx, did in enumerate(list_article_id)}

        flat_splits = []
        didx2sidx = []
        for did, text in enumerate(list_context_cleaned):
            splits = self._split_doc_slicing_window(text, size=120, stride=5)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        # 2. Tokenenize query and context with a tokenizer (default: newmm)
        q_tokens = self.processes.map_async(tokenize_text, list_query_question)
        c_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        c_tokens = c_tokens.get()
        print('[Done Tokenization]')

        
        # 3. Create structed example input
    
        examples = []
        for qidx in range(len(list_query_question)):
            for rel_did, did in enumerate(list_article_id):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                        if (len(q_tokens[qidx].words()) > 0 and
                                len(c_tokens[sidx].words()) > 0 and
                                rel_did == qidx):

                            examples.append({
                                'id': (qidx, rel_did, sidx),
                                'question': q_tokens[qidx].words(),
                                'qlemma': q_tokens[qidx].lemmas(),
                                'document': c_tokens[sidx].words(),
                                'lemma': c_tokens[sidx].lemmas(),
                                'pos': c_tokens[sidx].pos(),
                                'ner': c_tokens[sidx].entities(),
                            })
        
        print('[DONE Adding example of ', len(examples), ']')
        # 4. Predict the example

        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_loader(examples, num_loaders):
            # if candidates or self.fixed_candidates:
            #     batch_cands = []
            #     for ex_id in batch[-1]:
            #         batch_cands.append({
            #             'input': s_tokens[ex_id[2]],
            #             'cands': candidates[ex_id[0]] if candidates else None
            #         })
            #     handle = self.reader.predict(
            #         batch, batch_cands, async_pool=self.processes
            #     )
            # else:
            handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))



       
        # 5. Select top score answer
        queues = [[] for _ in range(len(list_query_question))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)


        # 6. Store prediction
        all_predictions = []

        predictions = []
        for queue in queues:
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)

                res = ner.get_ner(list_title[qidx], postag=False)
                # print('ner',res)

                for ent in res:
                    title_type = ent[1]
                    if title_type == 'O':
                        title_type = '-'
                    else:
                        title_type = title_type.split('-')[1]
                        break
                
                for ans in predictions:
                    if (ans['question_id'] == list_query[qidx]['question_id']):

                        ans['answer'].insert(0, c_tokens[sidx].slice(s, e + 1).untokenize())
                        ans['answer_begin_position'].insert(0, c_tokens[sidx].offsets()[s][0] + int(list_pad[qidx]) + 1),
                        ans['answer_end_position'].insert(0, c_tokens[sidx].offsets()[e][1] + int(list_pad[qidx]) + 1),
                        ans['answer_score'].insert(0, float(score * 100))
                        ans['context_window'].insert(0, c_tokens[sidx].untokenize())
                        continue

                prediction = {
                    'question_id': list_query[qidx]['question_id'],
                    'question': list_query[qidx]['question'],

                    'answer': [c_tokens[sidx].slice(s, e + 1).untokenize()],

                    # 'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    # 'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'answer_score': [float(score * 100) ],
                    'text': c_tokens[sidx].untokenize(),
                    'answer_begin_position': [c_tokens[sidx].offsets()[s][0] + int(list_pad[qidx]) + 1],
                    'answer_end_position': [c_tokens[sidx].offsets()[e][1] + int(list_pad[qidx]) + 1],
                    'article_id': list_article_id[qidx],
                    'title': list_title[qidx],
                    'title_type': title_type,
                    'context_window': [c_tokens[sidx].untokenize()],
                    'context_cleaned': list_context_cleaned[qidx],
                    'context_uncleaned': list_context_uncleaned[qidx]
                    # 'list_images_url': list_images_url[qidx]
                }

                predictions.append(prediction)
        # all_predictions.append(predictions[-1::-1])

        for qidx, title in enumerate(list_title):
            if title == None:
                prediction = {
                    'question_id': list_query[qidx]['question_id'],
                    'question': list_query[qidx]['question'],
                    'answer': None,
                    'answer_score': None,
                    'text': None,
                    'answer_begin_position': None,
                    'answer_end_position': None,
                    'article_id': None,
                    'title': None,
                    'title_type': None,
                    'context_window': None,
                    'context_cleaned': None,
                    'context_uncleaned': None
                    # 'list_images_url': None
                }
                predictions.append(prediction)

        # print(all_predictions)
        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))

        print('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        return predictions


    def process_batch_official(self, queries, candidates=None, top_n=1, n_docs=1, reutrn_context=False):
        """Run a single query"""
        '''
            Params:
                queries: is a list of dict {question_id: <str>, question: <str>}

            Return:
                question_id: as param queries
                question: as param queries
                answer: answer to the question
                start_position: start position based on UNCLEANED Wiki article
                end_position: end position based on UNCLEANED Wiki article
                context: context of UNCLEAN Wiki article
        '''
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)



        # 1. Retrieve the context of the wikipedia article in cleaned format (exclude <document> tag)
        list_query = []
        list_article_id = []
        list_context_cleaned = []
        list_context_uncleaned = []
        list_snippet = []
        list_pad = []
        list_query_question = []
        list_title = []

        for query in queries: 
            article_id, title, context, context_cleaned, snippet, pad = getContextFromQuestion(query['question'], image=False)
            if (article_id == 'not found'):
                print('article not found')
                list_query.append(query)
                list_query_question.append(query['question'])
                list_title.append(None)
                list_context_cleaned.append('')
                list_article_id.append(None)
                list_context_uncleaned.append('')
                list_snippet.append('')
                list_pad.append(0)
                continue

            list_query.append(query)
            list_query_question.append(query['question'])
            list_title.append(title)
            list_context_cleaned.append(context_cleaned)
            list_article_id.append(article_id)
            list_context_uncleaned.append(context)
            list_snippet.append(snippet)
            list_pad.append(pad)

        
        print('[DONE Fetching Document]')
        # did2didx = {did: didx for didx, did in enumerate(list_context_cleaned)}
        did2didx = {did: didx for didx, did in enumerate(list_article_id)}

        flat_splits = []
        didx2sidx = []
        for did, text in enumerate(list_context_cleaned):
            splits = self._split_doc_slicing_window(text, size=120, stride=5)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        # 2. Tokenenize query and context with a tokenizer (default: newmm)
        q_tokens = self.processes.map_async(tokenize_text, list_query_question)
        c_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        c_tokens = c_tokens.get()
        print('[Done Tokenization]')

        
        # 3. Create structed example input
    
        examples = []
        for qidx in range(len(list_query_question)):
            for rel_did, did in enumerate(list_article_id):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                        if (len(q_tokens[qidx].words()) > 0 and
                                len(c_tokens[sidx].words()) > 0 and
                                rel_did == qidx):

                            examples.append({
                                'id': (qidx, rel_did, sidx),
                                'question': q_tokens[qidx].words(),
                                'qlemma': q_tokens[qidx].lemmas(),
                                'document': c_tokens[sidx].words(),
                                'lemma': c_tokens[sidx].lemmas(),
                                'pos': c_tokens[sidx].pos(),
                                'ner': c_tokens[sidx].entities(),
                            })
        
        print('[DONE Adding example of ', len(examples), ']')
        # 4. Predict the example

        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_loader(examples, num_loaders):
            handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))



       
        # 5. Select top score answer
        queues = [[] for _ in range(len(list_query_question))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)


        # 6. Store prediction
        all_predictions = []

        predictions = []
        for queue in queues:
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)

                res = ner.get_ner(list_title[qidx], postag=False)
                # print('ner',res)

                for ent in res:
                    title_type = ent[1]
                    if title_type == 'O':
                        title_type = '-'
                    else:
                        title_type = title_type.split('-')[1]
                        break

                prediction = {
                    'question_id': int(list_query[qidx]['question_id']),
                    'question': list_query[qidx]['question'],
                    'answer': c_tokens[sidx].slice(s, e + 1).untokenize(),
                    'answer_begin_position': c_tokens[sidx].offsets()[s][0] + int(list_pad[qidx]) + 1,
                    'answer_end_position': c_tokens[sidx].offsets()[e][1] + int(list_pad[qidx]) + 1,
                    'article_id': int(list_article_id[qidx]),
                }

                predictions.append(prediction)
        # all_predictions.append(predictions[-1::-1])
        for qidx, title in enumerate(list_title):
            if title == None:
                prediction = {
                    'question_id': list_query[qidx]['question_id'],
                    'question': list_query[qidx]['question'],
                    'answer': None,
                    'answer_begin_position': None,
                    'answer_end_position': None,
                    'article_id': None,
                }
                predictions.append(prediction)

     
        # print(all_predictions)
        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))

        print('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        return predictions


