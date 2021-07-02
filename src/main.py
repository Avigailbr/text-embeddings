import sys
from elasticsearch import Elasticsearch

# Use tensorflow 1 behavior to match the Universal Sentence Encoder
# examples (https://tfhub.dev/google/universal-sentence-encoder/2).
import tensorflow as tf
import tensorflow_hub as hub

from elastic import index_data, handle_embedding_query, handle_text_query
from iteration import create_iteration

GPU_LIMIT = 0.5


if __name__ == '__main__':
    print("Downloading pre-trained embeddings from tensorflow hub...")
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    text_ph = tf.placeholder(tf.string)
    embeddings = embed(text_ph)
    print("Done.")

    print("Creating tensorflow session...")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = GPU_LIMIT
    session = tf.Session(config=config)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("Done.")

    es_client = Elasticsearch()

    #index_data(es_client=es_client, embeddings=embeddings, session=session, text_ph=text_ph)

    while True:
        try:
            val = input( """Enter:
            1)  for querying the collection through word-embedding similarity match
            2) for querying the collection through textual match
            3)  for creating the evaluation data 
            4)  to exist: """)

            if val == "1":
                handle_embedding_query(es_client, embeddings, session, text_ph)
            if val == "2":
                handle_text_query(es_client)
            if val == "3":
                create_iteration(es_client, embeddings, session, text_ph)
            if val == "4":
                break
            print("wrong input!")
        except KeyboardInterrupt as err:
            print("Unexpected error:", err)
            break

    print("Closing tensorflow session...")
    session.close()
    print("Done.")
