import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
#from umap.umap_ import UMAP
#import umap
import matplotlib.pyplot as plt
from collections import Counter

import pandas as pd
import ablang2 #import Tokenizer
import torch

import umap.umap_ as umap
#print(UMAP.__version__)

# Step 1: Define a function to convert sequences into k-mer frequency vectors
def kmer_encoding(sequence, k=3, kmer_set=None):
    """ Convert sequence to k-mer frequency vector, using a predefined kmer_set """
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    kmer_count = Counter(kmers)
    
    # Initialize the k-mer vector based on the fixed k-mer set
    kmer_vector = np.zeros(len(kmer_set))
    
    # Fill the vector with the counts for the k-mers
    for kmer, count in kmer_count.items():
        if kmer in kmer_set:
            kmer_vector[kmer_set[kmer]] = count
    
    return kmer_vector

# Step 1: Load the paired sequences from the CSV file
#get sequence from somewhere like the following
#https://opig.stats.ox.ac.uk/webapps/plabdab/

#df = pd.read_csv('sanitized_sequences.csv')
# Load the sequences (assumed to be in 'paired_sequences.csv')
df = pd.read_csv('paired_sequences.csv')

# Initialize the ablang2 model
ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=12, device='cuda')

# Prepare antibody sequences from the CSV
heavy_sequences = df['heavy_sequence'].tolist()
light_sequences = df['light_sequence'].tolist()

# Function to sanitize sequences by replacing invalid characters 'B' and 'Z' with '*'
def sanitize_sequence(seq):
    # Replace 'B' and 'Z' with '*'
    seq = seq.replace('B', '*')
    seq = seq.replace('Z', '*')
    return seq

# Check function to verify that no 'B' or 'Z' remain in the sequence
def check_invalid_characters(sequences):
    invalid_found = False
    for seq in sequences:
        if 'B' in seq or 'Z' in seq:
            invalid_found = True
            print(f"Invalid characters found in sequence: {seq}")
    return invalid_found

# Step 4: Sanitize the sequences to replace 'B' and 'Z' with '*'
sanitized_heavy_sequences = [sanitize_sequence(seq) for seq in heavy_sequences]
sanitized_light_sequences = [sanitize_sequence(seq) for seq in light_sequences]

# Step 5: Check if any invalid characters remain in the sanitized sequences
if check_invalid_characters(sanitized_heavy_sequences) or check_invalid_characters(sanitized_light_sequences):
    print("Warning: Some sequences still contain invalid characters ('B' or 'Z').")
else:
    print("All sequences have been sanitized successfully.")


# The format requires | to separate the VH and VL
#seqs = [f"{heavy}|{light}" for heavy, light in zip(sanitized_heavy_sequences, sanitized_light_sequences)]
seqs = [[heavy, light] for heavy, light in zip(sanitized_heavy_sequences, sanitized_light_sequences)]
#print(seqs)

# Step 6: Use the ablang2 restore function to restore the sequences
restored_sequences = [ablang(seq, mode='restore') for seq in seqs]

# Step 4: Save restored sequences as a new CSV file
restored_df = pd.DataFrame({
    'restored_sequence': restored_sequences,
})

# Save the restored sequences to a new CSV file
restored_df.to_csv('restored_sequences.csv', index=False)

# Step 6: Tokenize the sanitized sequences using ablang2 tokenizer

# Print the sequences that will be tokenized for debugging
#print("Sequences to be tokenized:", seqs)


#seqs = [f"{heavy}|{light}" for heavy, light in zip(sanitized_heavy_sequences, sanitized_light_sequences)]

# Tokenize the sequences
tokenized_seq = ablang.tokenizer(restored_sequences, pad=True, w_extra_tkns=False, device="cuda")
# Step 5: Generate k-mers (k=3) from both the heavy and light sequences
k = 3
all_kmers = set()


# Process heavy sequences to extract k-mers
for seq in sanitized_heavy_sequences:
    all_kmers.update([seq[i:i+k] for i in range(len(seq)-k+1)])

# Process light sequences to extract k-mers
for seq in sanitized_light_sequences:
    all_kmers.update([seq[i:i+k] for i in range(len(seq)-k+1)])

# Create a fixed k-mer set and index them for feature extraction
kmer_set = {kmer: idx for idx, kmer in enumerate(all_kmers)}

# Function to encode sequences as k-mer vectors
def kmer_encoding(seq, k, kmer_set):
    # Create a vector that indicates the presence of each k-mer in the sequence
    kmer_vector = np.zeros(len(kmer_set))
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_set:
            kmer_vector[kmer_set[kmer]] = 1
    return kmer_vector


# Step 5: Encode all sequences into k-mer vectors using the fixed k-mer set
sequence_vectors = np.array([kmer_encoding(seq, k=k, kmer_set=kmer_set) for seq in sanitized_heavy_sequences + sanitized_light_sequences])

# Step 6: Standardize the k-mer vectors (important for clustering)
scaler = StandardScaler()
sequence_vectors_scaled = scaler.fit_transform(sequence_vectors)

# Step 7: Apply UMAP for dimensionality reduction (2D for visualization)
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(sequence_vectors_scaled)

# Step 8: Apply DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=2)  # Parameters for DBSCAN; adjust based on data
dbscan_labels = dbscan.fit_predict(umap_embeddings)

# Step 9: Visualize the UMAP embeddings with DBSCAN clusters
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=dbscan_labels, cmap='viridis', s=100, edgecolors='k')
plt.title("UMAP + DBSCAN Clustering of Antibody Sequences")
plt.colorbar(label='Cluster ID')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Step 10: Print clustering results
for idx, label in enumerate(dbscan_labels):
    print(f"Sequence: {sanitized_heavy_sequences[idx]} \t Cluster: {label}")

# Step 11: Optionally, allow clustering to pick up specific sequences of interest
def find_sequences_with_kmer(sequences, kmer_pattern):
    """ Identify sequences containing a specific k-mer pattern """
    return [seq for seq in sequences if kmer_pattern in seq]

# Example: Find sequences with a specific k-mer ("CAG")
sequences_with_cag = find_sequences_with_kmer(sanitized_heavy_sequences + sanitized_light_sequences, "CAG")
print("\nSequences with 'CAG' k-mer:", sequences_with_cag)
