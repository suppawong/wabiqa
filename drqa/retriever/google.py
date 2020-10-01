import wikipedia
import requests
import json
from urllib.parse import unquote
from google import google
import glob
import os
import re
from .. import DATA_DIR

wikipedia.set_lang('th')
wikipedia_article_folder_path = os.path.join(DATA_DIR, 'wikipedia_articles')
articles_path = glob.glob(wikipedia_article_folder_path + '*.txt')
tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')


def cleanTextWithPad(text):
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    
    pad = len(tag_re.split(text)[1])
    return tag_re.sub('', text), pad

def removeXMLTag(text):
    return tag_re.sub('', text)

def getTitleOfDocument(text):
    return re.findall(r'<*title=\"(.+)\">',text)

def getDocumentContext(article_id):
    path = wikipedia_article_folder_path + '/' + str(article_id) + '.txt'
    # print(path)
    with open(path, 'r', encoding='utf-8') as f:
        context = f.read()

        context_cleanned, pad = cleanTextWithPad(context)

    return context, context_cleanned, pad
        
def searchGoogle(query, num_page):
    search_results = google.search(query, num_page)
    return search_results

def getFirstSearchResults(results, filter_domain=None):
    for result in results:
        link_split = []
        try:
            link_split = result.link.split('/')
        except:
            continue

        if filter_domain in link_split:
            link = link_split[-1]
            snippet = result.description
            return unquote(link), snippet

    link_split = []
    try:
        firstResult = results[0]
        link = firstResult.link.split('/')[-1]
        snippet = firstResult.description
        return unquote(link), snippet
    except:
        pass
    return 'NA', 'NA'
   

def replace(text, before, after):
    return text.replace(before,after)

def getDocument(article_id):
    # return document context (uncleaned) givenn document ids
    return ''


def searchWikiArticle(q, search_space=1):
    '''
        Params
            q: question in Thai language
        Return
            data: Wikipedia data
            snippet: snippet text from Google Search 
    '''
    # given query
    # return wiki_page
    
    # search via google
    results = searchGoogle(q, search_space)
    # print(results)
    link, snippet = getFirstSearchResults(results, filter_domain='th.wikipedia.org')

    # print(link, snippet)
    # search for wiki page id based of result found on google
    s = requests.Session()
    # add header to to trick browser
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = "https://th.wikipedia.org/w/api.php"
    params = {
        'action':"query",
        'prop':"info",
        'titles':link,
        'format':"json"
    }
    r = s.get(url=url, params=params, headers=headers)
    try:
        data = r.json()['query']['pages']
        print(data)
        wikipedia_article_id = list(data.keys())[0]
        print('wikipedia_article_id',wikipedia_article_id)
        title = data[wikipedia_article_id]['title']
        print('title',title)

        context, context_cleaned, pad = getDocumentContext(wikipedia_article_id)
        # print('context', context, context_cleaned)

        # print(r.json()['query'])

        params = {
            'action':"query",
            'prop':"images",
            'titles':link,
            'format':"json"
        }
        r = s.get(url=url, params=params, headers=headers)

        return wikipedia_article_id, title, context, context_cleaned, snippet, pad

    except:
        return "not found","not found","not found","not found","not found","not found"


def searchWikiArticleImage(q, search_space=1):
    '''
        Params
            q: question in Thai language
        Return
            data: Wikipedia data
            snippet: snippet text from Google Search 
    '''
    # given query
    # return wiki_page
    
    # search via google
    results = searchGoogle(q, search_space)
    # print(results)
    link, snippet = getFirstSearchResults(results, filter_domain='th.wikipedia.org')

    # print(link, snippet)
    # search for wiki page id based of result found on google
    s = requests.Session()
    # add header to to trick browser
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = "https://th.wikipedia.org/w/api.php"
    params = {
        'action':"query",
        'prop':"info",
        'titles':link,
        'format':"json"
    }
    r = s.get(url=url, params=params, headers=headers)
    try:
        data = r.json()['query']['pages']
        print(data)
        wikipedia_article_id = list(data.keys())[0]
        print('wikipedia_article_id',wikipedia_article_id)
        title = data[wikipedia_article_id]['title']
        print('title',title)

        context, context_cleaned, pad = getDocumentContext(wikipedia_article_id)
        # print('context', context, context_cleaned)

        # print(r.json()['query'])

        params = {
            'action':"query",
            'prop':"images",
            'titles':link,
            'format':"json"
        }
        r = s.get(url=url, params=params, headers=headers)
        


        try:
            data = r.json()['query']['pages']
            print('QUERY: Image', r.json()['query'])

            wikipedia_article_id = list(data.keys())[0]
            images = data[wikipedia_article_id]['images']
            
            images_title = []
            for img in images:
                images_title.append(img['title'])
            print(images_title)

            images_url = []
            for title in images_title:
                res = get_image_info(title)
                print(res)
                if (res != None):
                    images_url.append(res)

            return wikipedia_article_id, title, context, context_cleaned, snippet, pad, images_url

        except:
            return wikipedia_article_id, title, context, context_cleaned, snippet, pad, []
    except:
        return "not found","not found","not found","not found","not found","not found", "not found"

def get_image_info(name):
    s = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    url = "https://th.wikipedia.org/w/api.php"
    params = {
        "action":"query",
        "format":"json",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": name,
    }
    r = s.get(url=url, params=params, headers=headers)
   
    try:
        data = r.json()['query']['pages']['-1']['imageinfo'][0]['url']
        return data
    except:
        # print('ERROR GET IMAGE INFO', name)
        return None
    


# def searchWikiArticle(q, search_space=1):
#     '''
#         Params
#             q: question in Thai language
#         Return
#             data: Wikipedia data
#             snippet: snippet text from Google Search 
#     '''
#     # given query
#     # return wiki_page
    
#     # search via google
#     results = searchGoogle(q, search_space)
#     # print(results)
#     link, snippet = getFirstSearchResults(results, filter_domain='th.wikipedia.org')

#     # print(link, snippet)
#     # search for wiki page id based of result found on google
#     s = requests.Session()
#     # add header to to trick browser
#     headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
#     url = "https://th.wikipedia.org/w/api.php"
#     params = {
#         'action':"query",
#         'prop':"info",
#         'titles':link,
#         'format':"json"
#     }
#     r = s.get(url=url, params=params, headers=headers)
#     try:
#         data = r.json()['query']['pages']
#         print(data)
#         wikipedia_article_id = list(data.keys())[0]
#         print('wikipedia_article_id',wikipedia_article_id)
#         title = data[wikipedia_article_id]['title']
#         print('title',title)

#         context, context_cleaned, pad = getDocumentContext(wikipedia_article_id)
#         # print('context', context, context_cleaned)
#         return wikipedia_article_id, title, context, context_cleaned, snippet, pad
#     except:
#         return "not found","not found","not found","not found","not found","not found"

