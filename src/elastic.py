import json
import time

from elasticsearch.helpers import bulk

from src.embed import embed_text

INDEX_NAME = "posts"
INDEX_FILE = "../data/posts/index.json"

DATA_FILE = "../data/posts/posts.json"

BATCH_SIZE = 1000
SEARCH_SIZE = 4


def index_data(es_client, embeddings, session, text_ph):
    print("Creating the 'posts' index.")
    es_client.indices.delete(index=INDEX_NAME, ignore=[404])

    with open(INDEX_FILE) as index_file:
        source = index_file.read().strip()
        es_client.indices.create(index=INDEX_NAME, body=source)

    docs = []
    count = 0

    with open(DATA_FILE) as data_file:
        for line in data_file:
            line = line.strip()

            doc = json.loads(line)
            if doc["type"] != "question":
                continue

            docs.append(doc)
            count += 1

            if count % BATCH_SIZE == 0:
                index_batch(es_client, docs, embeddings, session, text_ph)
                docs = []
                print("Indexed {} documents.".format(count))

        if docs:
            index_batch(es_client, docs, embeddings, session, text_ph)
            print("Indexed {} documents.".format(count))

    es_client.indices.refresh(index=INDEX_NAME)
    print("Done indexing.")


def index_batch(es_client, docs, embeddings, session, text_ph):
    titles = [doc["title"] for doc in docs]
    title_vectors = embed_text(session, embeddings, text_ph, titles)

    requests = []
    for i, doc in enumerate(docs):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = INDEX_NAME
        request["title_vector"] = title_vectors[i]
        requests.append(request)
    bulk(es_client, requests)


def handle_embedding_query(es_client, embeddings, session, text_ph):
    query = input("Enter query: ")

    embedding_start = time.time()
    query_vector = embed_text(session, embeddings, text_ph, [query])[0]
    embedding_time = time.time() - embedding_start

    search_start = time.time()
    response = embedded_search(es_client, query_vector)
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("embedding time: {:.2f} ms".format(embedding_time * 1000))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


def handle_text_query(es_client):
    query = input("Enter query: ")

    search_start = time.time()
    response = textual_search(es_client, query)
    search_time = time.time() - search_start

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    print("search time: {:.2f} ms".format(search_time * 1000))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()


def embedded_search(es_client, query_vector, doc_id=""):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, doc['title_vector']) + 1.0",
                "params": {"query_vector": query_vector}
            }
        },
    }

    response = es_client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": {"bool": {"must_not": [exclude_doc_id_query(doc_id)], "should": [script_query]}},
            "track_total_hits": True,
            "_source": {"includes": ["title", "body"]},
            "min_score": 1.5
        }
    )

    return response


def textual_search(es_client, query, doc_id=""):
    response = es_client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": {"bool": {"must_not": [exclude_doc_id_query(doc_id)], "should": [{"match": {"title": query}}]}},
            "track_total_hits": True,
            "_source": {"includes": ["title", "body"]}
        }
    )

    return response


def exclude_doc_id_query(doc_id):
    return {"term": {"questionId": {"value": doc_id}}}
