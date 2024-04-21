from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from nltk.tokenize import sent_tokenize

from file_reader import read_file  # Assuming file_reader.py contains your read_file function

# Load the BERT model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to encode a sentence
def encode_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to extract bullet points
def extract_bullet_points(text, num_points):
    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Encode each sentence
    sentence_vectors = [encode_sentence(sentence) for sentence in sentences]

    # Perform clustering
    kmeans = KMeans(n_clusters=num_points)
    kmeans.fit(np.concatenate(sentence_vectors))

    # Select a representative sentence for each cluster
    bullet_points = []
    for i in range(num_points):
        cluster_center = kmeans.cluster_centers_[i]
        closest_sentence_index = np.argmin([np.linalg.norm(vec - cluster_center) for vec in sentence_vectors])
        bullet_points.append(sentences[closest_sentence_index])

    return bullet_points

# Read the text from a file
text = read_file(r"D:\\MyWork\\toolongtoread.com\\filetomark.txt")

# Extract bullet points
bullet_points = extract_bullet_points(text, num_points=3)
for point in bullet_points:
    print(f"- {point}")