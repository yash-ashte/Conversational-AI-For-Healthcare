from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import sys, os
cd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cd, '..')))
from chatbot import Chatbot

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus with example sentences
with open("Generations/questions.txt", "r", encoding="utf-8") as file:
    corpus = file.read().strip().split("\n")

corpus_embeddings = embedder.encode(corpus)
num_clusters = 8

# Perform kmean clustering
clustering_model = KMeans(n_clusters=num_clusters, random_state=0)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]

dist = clustering_model.transform(corpus_embeddings)
clustered_dists = [[] for i in range(num_clusters)]
clustered_idx = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
    clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
    clustered_idx[cluster_id].append(sentence_id)

with open("Generations/Question_Clusters.txt", "w") as file:
    for cluster_id in range(num_clusters):
        file.write(f"Cluster {cluster_id}: {clustered_sentences[cluster_id]}\n")

question_list = []
for i in range(len(clustered_dists)):
    min_index = 0
    for j in range(1, len(clustered_dists[i])):
        if clustered_dists[i][j] < clustered_dists[i][min_index]:
            min_index = j
    question_list.append(clustered_sentences[i][min_index])

print(f"Representative questions of the {num_clusters} clusters for zero-shot generation")
with open("Generations/Representive_Questions.txt", "w") as file:
    for question in question_list:
        file.write(question + "\n")
        print(question)

chatbot = Chatbot()

with open("Prompts/CoT_Header.txt", "r") as file:
    content_procedure = file.read()

with open("Prompts/CoT_Response_Header.txt", "r") as file:
    response_procedure = file.read()

chatbot.load_CoT_Header(content_procedure)
chatbot.load_CoT_Response_Header(response_procedure)
output = chatbot.run(question_list[4])

with open("outputs/CoT_Output.txt", "w") as file:
    file.write(output)

