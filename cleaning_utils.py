
import hashlib
from typing import List, Dict, Union, Tuple
import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasketch
import multiprocessing

THRESHOLD = 0.7

def remove_duplicates(functions: List[Dict]):
    # assume dict is in form of uuid: str, input: str
    # make the assumption we take the function with the higher uuid
    unique_functions = {}
    for function in functions:
        hash = hashlib.sha256(function["input"].encode()).hexdigest()
        if hash not in unique_functions:
            unique_functions[hash] = function
        else:
            if function["uuid"] > unique_functions[hash]["uuid"]:
                unique_functions[hash] = function
    return list(unique_functions.values())


def create_minhashes(documents: List[Dict[str, str]], 
                    ngram_size: int = 5, bands: int = 20, rows_per_band: int = 128) -> Tuple[Dict[str, datasketch.MinHash], int]:
    """
    Create MinHash signatures for a list of documents with LSH bands configuration.
    
    Args:
        documents: List of dictionaries, each containing 'uuid' and 'input' keys
        num_permutations: Number of hash functions to use (default: 100)
        ngram_size: Size of n-grams to generate from input text (default: 3)
        bands: Number of bands for LSH (default: 20)
    
    Returns:
        Tuple containing:
        - Dictionary mapping document UUIDs to their MinHash signatures
        - Rows per band (num_permutations / bands)
    
    Raises:
        ValueError: If num_permutations is not divisible by bands
    """

    num_permutations = rows_per_band * bands
    
    
    def generate_ngrams(text: str, n: int) -> List[str]:
        """Generate n-grams from input text."""
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    
    # Initialize result dictionary
    minhash_dict = {}
    # Process each document
    for doc in tqdm(documents, desc="Creating minhashes"):
        minhash = datasketch.MinHash(num_perm=num_permutations)
        uuid = doc['uuid']
        text = doc['input'].lower()  # Convert to lowercase for consistency
        
        # Generate n-grams
        ngrams = generate_ngrams(text, ngram_size)
        for ngram in ngrams:
            minhash.update(ngram.encode('utf8'))
        
        minhash_dict[uuid] = minhash
    
    return minhash_dict


# 16 bands with 128 rows
def create_similarity_matrix(minhashes: Dict[str, datasketch.MinHash], rows_per_band: int, num_bands: int, threshold: float) -> np.ndarray:
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_bands*rows_per_band)
    print(f"num_perm: {num_bands*rows_per_band}")
    similarity_matrix = {}
    for uuid, minhash in tqdm(minhashes.items(), desc="Inserting minhashes into LSH"):
        lsh.insert(uuid, minhash)
    for uuid, minhash in tqdm(minhashes.items(), desc="Querying LSH"):
        similar_uuids = lsh.query(minhash)
        similarity_matrix[uuid] = similar_uuids
    for uuid, similar_uuids in tqdm(similarity_matrix.items(), desc="Removing self-similarities"):
        if uuid in similar_uuids:
            similar_uuids.remove(uuid)
    return similarity_matrix


def create_histogram_of_matrix(simarity_matrix: Dict[int, List[int]], filename: str="similarity_matrix_histogram.png"):
    # create a histogram of the similarity matrix
    plt.hist([len(similar_uuids) for similar_uuids in simarity_matrix.values()], 
             bins=20, density=True, weights=np.ones(len(simarity_matrix))/len(simarity_matrix))
    plt.savefig(filename)
    plt.close()

def filter_matrix(similarity_matrix: Dict[int, List[int]]) -> Dict[int, List[int]]:
    good_uuids = []
    for uuid, similar_uuids in similarity_matrix.items():
        good_uuids.append(uuid)
        for similar_uuid in similar_uuids:
            # tiebreak on largest uuid
            if uuid < similar_uuid:
                good_uuids.remove(uuid)
                break
    return good_uuids

def fuzzy_filter(documents: List[Dict], threshold: float=0.7, ngram_size: int=5, bands: int=16, rows_per_band: int=128, create_histogram: bool=False) -> List[Dict]:
    minhashes = create_minhashes(documents, ngram_size=ngram_size, bands=bands, rows_per_band=rows_per_band)
    similarity_matrix = create_similarity_matrix(minhashes, rows_per_band=rows_per_band, num_bands=bands, threshold=threshold) 
    if create_histogram:
        create_histogram_of_matrix(similarity_matrix, filename=f"similarity_matrix_histogram_{ngram_size}_{bands}_{rows_per_band}.png")
    good_uuids = filter_matrix(similarity_matrix)
    good_documents = [{'uuid': uuid, 'input': documents[uuid]['input']} for uuid in good_uuids]
    return good_documents

if __name__ == "__main__":
    with open("github_triton.json", "r") as f:
        data = json.load(f)
    documents = []
    count = 0
    # check for testing flag
    testing = False
    if testing:
        data = data[:10]
    for function in data:
        documents.append({"uuid": count, "input": function["input"]})
        count += 1
    
    unique_functions = remove_duplicates(documents)
    print(f"porportion of unique functions: {len(unique_functions)/len(documents)}")

    good_documents = fuzzy_filter(documents, threshold=0.7, ngram_size=5, bands=16, rows_per_band=128, create_histogram=True)
    
    # save cleaned data as json
    with open("github_triton_cleaned.json", "w") as f:
        json.dump(good_documents, f)

    
    
