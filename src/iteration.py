from csv import reader
from datetime import date
import time

from src.embed import embed_text
from src.elastic import embedded_search, textual_search

QUESTIONS_FILE = "../data/questions.csv"


def create_iteration(es_client, embeddings, session, text_ph):
    name = "iteration_%s.csv" % (date.today().strftime("%b-%d-%Y"))
    file = open(name, 'w')
    header = ['Query', 'First embedded title result', 'First embedded id', 'Second embedded title results',
              'Second embedded id', 'Threed embedded title result', 'Threed embedded id', 'Fourth embedded title result',
              'Fourth embedded id', 'Embedded total results', 'Embedded request time',
              'First textual title result', 'First textual id', 'Second textual title results',
              'Second textual id', 'Threed textual title result', 'Threed textual id', 'Fourth textual title result',
              'Fourth textual id', 'Textual total results', 'Textual request time']

    file.write('"' + '","'.join(header) + '"\n')

    with open(QUESTIONS_FILE, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            query = row[0]
            question_id = row[1]
            data = [query]

            # embedding search
            start_start = time.time()
            query_vector = embed_text(session, embeddings, text_ph, [query])[0]

            response = embedded_search(es_client, query_vector, question_id)

            search_time = "{:.2f} ms".format((time.time() - start_start) * 1000)
            total_results = str(response["hits"]["total"]["value"])
            hits = []
            for hit in response["hits"]["hits"]:
                hits.append(hit["_source"]["title"])
                hits.append(hit["_id"])

            if len(hits) < 8:
                for i in range(8-len(hits)):
                    hits.append("")

            data = data + hits
            data.append(total_results)
            data.append(search_time)

            # textual search
            search_start = time.time()
            response = textual_search(es_client, query, question_id)

            search_time = "{:.2f} ms".format((time.time() - search_start) * 1000)
            total_results = str(response["hits"]["total"]["value"])
            hits = []
            for hit in response["hits"]["hits"]:
                hits.append(hit["_source"]["title"])
                hits.append(hit["_id"])

            if len(hits) < 8:
                for i in range(8 - len(hits)):
                    hits.append("")

            data = data + hits
            data.append(total_results)
            data.append(search_time)

            file.write('"' + '","'.join(data) + '"\n')

    file.close()
