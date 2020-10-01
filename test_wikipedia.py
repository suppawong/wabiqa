
from drqa import retriever

def testcase1_image():
    query = "ชวน หลีกภัย อดีตนายกรัฐมนตรีคนที่ 20 ของไทย เกิดเมื่อใด"
    res = retriever.google.searchWikiArticle(query)
    print(res)

def testcase2_image():
    query = "ใครคือนายกคนที่ 7 ของประเทศไทย"
    res = retriever.google.searchWikiArticle(query)
    # print(res)

if __name__ == '__main__':

    testcase2_image()