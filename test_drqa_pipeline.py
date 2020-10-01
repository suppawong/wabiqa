

from drqa import pipeline


def testcase1_single(pipeline):
    # TESTCASE 1:
    q = 'ชวน หลีกภัย อดีตนายกรัฐมนตรีคนที่ 20 ของไทย เกิดเมื่อวันที่เท่าไร'
    res = pipeline.process_single(q, 1)

    # print(b,e)
    print('begin,end = ', res['answer_begin_position'] ,res['answer_end_position'])
    print('EXCEPT begin = 293, end: 295')
    # print(res)
    context = res['context_window']
    print(context)

def testcase2_multiple(pipeline):

    queries = [{
        'question_id': '1',
        'question': 'ชวน หลีกภัย อดีตนายกรัฐมนตรีคนที่ 20 ของไทย เกิดเมื่อวันที่เท่าไร'
    },{
        'question_id': '2',
        'question': 'ยิ่งลักษณ์ ชินวัตร อดีตนายกรัฐมนตรีคนที่ 28 ของไทย เกิดวันที่เท่าไร'
    }]

    '''
    {'question_id': 2317, 'question': 'ชวน หลีกภัย อดีตนายกรัฐมนตรีคนที่ 20 ของไทย เกิดเมื่อวันที่เท่าไร',
     'answer': '28', 'answer_begin_position ': 293, 'answer_end_position': 295, 'article_id': 6130}

    {'question_id': 2326, 'question': 'ยิ่งลักษณ์ ชินวัตร อดีตนายกรัฐมนตรีคนที่ 28 ของไทย เกิดวันที่เท่าไร', 
    'answer': '21', 'answer_begin_position ': 141, 'answer_end_position': 143, 'article_id': 63252}
    '''
    res = pipeline.process_batch(queries)
    
    for qa in res:
        print('[Predicted]')
        print(qa['question_id'],qa['question'])
        print('answer', qa['answer'], qa['answer_begin_position'], qa['answer_end_position'],'score:', qa['answer_score'])
        if qa['question_id'] == '1':
            # assert(qa['answer'], '28')
            print('Groundtruh answer = 28', 'b293,e295')
        if qa['question_id'] == '2':
            # assert(qa['answer'], '21')
            print('Groundtruh answer = 21', 'b141,e143')
        print(qa['context_window'])
        print('')

def testcase3_multiple(pipeline):

    queries = [{
        'question_id': '1',
        'question': 'sdadsa'
    },{
        'question_id': '2',
        'question': 'asdsad'
    }]

   
    res = pipeline.process_batch(queries)
    
    for qa in res:
        print('[Predicted]')
        print(qa['question_id'],qa['question'])
        print('answer', qa['answer'], qa['answer_begin_position'], qa['answer_end_position'],'score:', qa['answer_score'])
        if qa['question_id'] == '1':
            assert(qa['answer'], None)
        if qa['question_id'] == '2':
            assert(qa['answer'], None)
        print(qa['context_window'])
        print('')
       
      
def testcase3_multiple_official(pipeline):

    queries = [{
        'question_id': '1',
        'question': 'sdadsa'
    },{
        'question_id': '2',
        'question': 'ยิ่งลักษณ์ ชินวัตร อดีตนายกรัฐมนตรีคนที่ 28 ของไทย เกิดวันที่เท่าไร'
    }]

   
    res = pipeline.process_batch_official(queries)
    
    for qa in res:
        print('[Predicted]')
        print(qa['question_id'],qa['question'])
        print('answer', qa['answer'], qa['answer_begin_position'], qa['answer_end_position'])
        if qa['question_id'] == '1':
            assert(qa['answer'], None)
        if qa['question_id'] == '2':
            assert(qa['answer'], '21')
        # print(qa['context_window'])
        print('')
       

if __name__ == '__main__':
    # context = getContextFromQuestion('ใครคือนายกรัญมนตรีคนที่ 7')
    
    # p = pipeline.create_drqa_instance('pilot')

    print('\nmodel: pilot \n')

    p = pipeline.create_drqa_instance('pilot')

    testcase3_multiple_official(p)

    # print('\nmodel: saiko.4\n')
    # p = pipeline.create_drqa_instance('saiko.4')

    # testcase2_multiple(p)

    # print('\nmodel: saiko.2\n')
    # p = pipeline.create_drqa_instance('saiko.2')

    # testcase2_multiple(p)
    # testcase2_multiple(p)

