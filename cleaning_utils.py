
import hashlib
from typing import List, Dict, Union, Tuple
import numpy as np
from collections import defaultdict

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

def LHS_clean(functions: List[Dict], threshold: float = 0.7, ngram_size: int = 5, ):
    # remove fuzzy duplicates using lhs
    function_ngrams = {}
    vocab = set()
    for function in functions:
        input = function["input"]
        function_ngrams[function["uuid"]] = set([input[i:i+ngram_size] for i in range(len(input) - ngram_size + 1)])
        for ngram in function_ngrams[function["uuid"]]:
            vocab.add(ngram)
    
    vocab = list(vocab)

    one_hot_vectors = {function_ngram["uuid"]: [1 if ngram in function_ngrams[function_ngram["uuid"]] else 0 for ngram in vocab] for function_ngram in function_ngrams}



    functions = remove_duplicates(functions)
    return functions



def create_minhashes(documents: List[Dict[str, str]], num_permutations: int = 100, 
                    ngram_size: int = 3, bands: int = 20) -> Tuple[Dict[str, np.ndarray], int]:
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
    if num_permutations % bands != 0:
        raise ValueError(f"Number of permutations ({num_permutations}) must be divisible by number of bands ({bands})")
    
    rows_per_band = num_permutations // bands
    
    def generate_ngrams(text: str, n: int) -> List[str]:
        """Generate n-grams from input text."""
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def hash_function(x: str, seed: int) -> int:
        """Create a hash value for a string using a seed."""
        return int(hashlib.md5(f"{seed}{x}".encode()).hexdigest(), 16)
    
    # Initialize result dictionary
    minhash_dict = {}
    
    # Process each document
    for doc in documents:
        uuid = doc['uuid']
        text = doc['input'].lower()  # Convert to lowercase for consistency
        
        # Generate n-grams
        ngrams = generate_ngrams(text, ngram_size)
        
        # Initialize minhash signature array
        signature = np.full(num_permutations, np.inf)
        
        # Generate minhash signature
        for i in range(num_permutations):
            # Calculate hash values for all n-grams using current permutation
            hash_values = [hash_function(ngram, i) for ngram in ngrams]
            # Store minimum hash value
            if hash_values:  # Check if there are any hash values
                signature[i] = min(hash_values)
        
        minhash_dict[uuid] = signature
    
    return minhash_dict, rows_per_band

def get_band_hashes(signature: np.ndarray, rows_per_band: int) -> List[int]:
    """
    Convert a signature into band hashes for LSH.
    
    Args:
        signature: MinHash signature array
        rows_per_band: Number of rows per band
    
    Returns:
        List of hash values for each band
    """
    bands = len(signature) // rows_per_band
    band_hashes = []
    
    for i in range(bands):
        start_idx = i * rows_per_band
        end_idx = start_idx + rows_per_band
        band = signature[start_idx:end_idx]
        # Hash the band values together
        band_hash = hash(tuple(band))
        band_hashes.append(band_hash)
    
    return band_hashes

# 16 bands with 128 rows
def calculate_band_similarity(band_hashes1: List[int], band_hashes2: List[int]) -> float:
    """
    Calculate the Jaccard similarity between two sets of band hashes.
    
    Args:
        band_hashes1: First list of band hashes
        band_hashes2: Second list of band hashes
    
    Returns:
        Jaccard similarity between the band hashes (float between 0 and 1)
    
    Raises:
        ValueError: If band hash lists have different lengths
    """
    if len(band_hashes1) != len(band_hashes2):
        raise ValueError("Band hash lists must have the same length")
    
    # Convert lists to sets for set operations
    set1 = set(band_hashes1)
    set2 = set(band_hashes2)
    
    # Calculate intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return intersection / union


def create_band_hashes(documents: List[Dict[str, str]], num_rows_per_band: int =128, ngram_size: int = 5, bands: int = 16) -> List[Dict[str, str]]:
    minhashes, rows_per_band = create_minhashes(documents, num_permutations=bands*num_rows_per_band, bands=20)
    band_hashes = []
    for uuid, signature in minhashes.items():
        band_hashes.append({"uuid": uuid, "band_hashes": get_band_hashes(signature, rows_per_band)})
    return band_hashes

def create_similarity_matrix(band_hashes: List[Dict[str, str]]) -> np.ndarray:
    similarity_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(band_hashes)):
        for j in range(i+1, len(band_hashes)):
            similarity_matrix[band_hashes[i]["uuid"]][band_hashes[j]["uuid"]] = calculate_band_similarity(band_hashes[i]["band_hashes"], band_hashes[j]["band_hashes"])
    return similarity_matrix

if __name__ == "__main__":
# Example usage
    documents = [
        {'uuid': '123', 'input': 'This is the first document'},
        {'uuid': '456', 'input': 'This is another document'},
        {'uuid': '789', 'input': 'Something completely different'}
    ]
    # Create minhashes with 100 permutations divided into 20 bands
    minhashes, rows_per_band = create_minhashes(documents)

    band_hashes = create_band_hashes(documents)
    similarity_matrix = create_similarity_matrix(band_hashes)
    print(similarity_matrix)