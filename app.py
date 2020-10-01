from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import json
import drqa

# from wabiqa.retriever import searchWikiArticle
# from wabiqa.pipeline import create_drqa_instance

app = Flask(__name__)
api = Api(app)

p = drqa.pipeline.create_drqa_instance('saiko.6')

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class DocumentRetriever(Resource):
    def post(self):
        args = parser.parse_args()
        question = args['question']

        # TODO
        # Send To retriever
        wikipedia_article_id, title, context, context_cleaned, snippet, pad, images_url = drqa.retriever.searchWikiArticle(question)

       
        return {
            'article_id': wikipedia_article_id, 
            'title': title, 
            'context': context,
            'context_cleaned': context_cleaned, 
            'snippet': snippet,
            'pad': pad,
            'images_url': images_url
        }

class QuerySingle(Resource):
    def post(self):


        args = parser.parse_args()
        question = args['question']

        res = p.process_single(question, 1)

        return res

class QueryMultipleOfficial(Resource):
    def post(self):


        # args['data'] contains a list of (question_id, question)
        queries  = request.get_json()
        # print(queries['data'])


        res = p.process_batch_official(queries['data'],top_n=1)
        obj = {
            "data": res
        }
        return obj

parser = reqparse.RequestParser()
parser.add_argument('question')
parser.add_argument('data')

api.add_resource(HelloWorld, '/')
api.add_resource(DocumentRetriever, '/retrieve')
api.add_resource(QuerySingle, '/query')
api.add_resource(QueryMultipleOfficial, '/query_official')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=80)
