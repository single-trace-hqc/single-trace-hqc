import ctypes
import logging
import numpy as np
import random
import matplotlib
import json
import math
from pathlib import Path
from scipy.sparse import identity, hstack, csr_matrix, coo_matrix
import os
import hashlib



def generate_random_binary_vector(length, weight):
    """Generates a random binary vector with length length and weight weight."""
    if not 0 <= weight <= length:
        raise ValueError("Weight must be between 0 and length")
    vector = np.zeros(length, dtype=np.int32)
    if weight > 0:
        indices = random.sample(range(length), weight)
        vector[indices] = 1
    return vector


def sparse_circulant(c):
    n = len(c)
    nz_rows = np.flatnonzero(c)
    num_nz_per_col = len(nz_rows)
    total_nz = num_nz_per_col * n

    col_indices = np.arange(n).repeat(num_nz_per_col)
    row_indices = (np.arange(n)[:, None] + nz_rows[None, :]) % n
    row_indices = row_indices.ravel()

    data = np.ones(total_nz, dtype=np.int32)
    return coo_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.int32).tocsr()


def apply_bit_flip(vector, rho):
    """Flips bits in a binary vector with a given probability rho."""
    flipped_vector = vector.copy()
    flip_mask = np.random.rand(len(flipped_vector)) < rho
    flipped_vector[flip_mask] = 1 - flipped_vector[flip_mask]
    return flipped_vector

def calculate_num_errors_first_n(n,array):
    half_sorted_xy = array[:n]
    return np.sum(half_sorted_xy)

def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    return lib

def load_config(scheme):
    with open('./src/config.json', 'r') as file:
        config = json.load(file)
    return config.get(scheme, {})

def setup_logging(script,scheme):
    # Constants
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    # Remove all handlers associated with the root logger (clean up)
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True) 
    log_file_path = log_dir / f'{script}_{scheme}.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up new logging configuration
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[
        logging.FileHandler(log_file_path),
        #logging.StreamHandler()
    ])
    # Set Matplotlib's logger to only log warnings or higher
    matplotlib_logger = logging.getLogger(matplotlib.__name__)
    matplotlib_logger.setLevel(logging.WARNING)


scheme_data = {
    'hqc1': {'n2': 384, 'probabilities': [23.44, 34.38, 24.83, 11.77, 4.12, 1.45]},
    'hqc3': {'n2': 640, 'probabilities': [16.50, 30.00, 27.00, 16.04, 7.07, 3.40]},
    'hqc5': {'n2': 640, 'probabilities': [23.14, 34.06, 24.87, 12.02, 4.32, 1.59]}
}

def sample_vector_from_scheme(scheme):
    if scheme not in scheme_data:
        raise ValueError(f"Scheme {scheme} not recognized. Available schemes are: {list(scheme_data.keys())}.")
    
    data = scheme_data[scheme]
    n2 = data['n2']
    probabilities = np.array(data['probabilities'])
    probabilities /= 100  # Convert percentages to a proper probability sum
    probabilities /= probabilities.sum()

    # Generate the number of ones in the vector
    number_of_ones = np.random.choice(np.arange(len(probabilities)), p=probabilities)

    # Create the binary vector
    vector = np.zeros(n2, dtype=int)
    vector[:number_of_ones] = 1
    np.random.shuffle(vector)
    
    return vector

def bit_array_to_uint64(vector):
    # Convert a binary array (LSB-first) into packed uint64 words for the C implementation
    len_64 = max(1, math.ceil(len(vector) / 64))
    uint64_values = []
    for word_index in range(len_64):
        word = 0
        base = word_index * 64
        for bit_offset in range(64):
            bit_index = base + bit_offset
            if bit_index < len(vector) and vector[bit_index] & 1:
                word |= 1 << bit_offset
        uint64_values.append(word)

    return (ctypes.c_uint64 * len_64)(*uint64_values)

def rm_decoder_result_wo_noise(eXORu,lib):
    # This function in C is added by the paper's authors to output the RM decoder result for one block
    rm_decoder = lib.reed_muller_decode_one_block
    rm_decoder.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    rm_decoder.restype = None

    vector_64 = bit_array_to_uint64(eXORu)
    message = (ctypes.c_uint64 * 1)()
    rm_decoder(message, vector_64)

    byte_view = ctypes.cast(message, ctypes.POINTER(ctypes.c_uint8))
    return int(byte_view[0])


def sample_binary_vector_in_weight_range(n2, min_weight, max_weight):
    """
    Sample a binary random vector of length n2 with Hamming weight between min_weight and max_weight.
    """
    # Calculate actual Hamming weight bounds
    lower_bound = int(min_weight * n2)
    upper_bound = int(max_weight * n2)
    
    # Generate a random Hamming weight within the bounds
    weight = random.randint(lower_bound, upper_bound)
    
    # Create the vector
    vector = np.zeros(n2, dtype=int)
    vector[:weight] = 1
    np.random.shuffle(vector)
    
    return vector

def mutate_bit_by_bit(vector): # only flip one bit
    flip_index = random.randint(0, len(vector) - 1)
    # Flip the bit: XOR with 1 will toggle the bit at flip_index
    vector[flip_index] ^= 1
    return vector

def mutate2bits(vector): # flip two bits

    flip_indices = random.sample(range(len(vector)), 2)

    for flip_index in flip_indices:
        vector[flip_index] ^= 1
    return vector

def mutate3bits(vector): # flip three bits

    flip_indices = random.sample(range(len(vector)), 3)

    for flip_index in flip_indices:
        vector[flip_index] ^= 1
    return vector

def hqc_entropy_for_key(key_idx, env_var="HQC_SEED_HEX", seed_len=32):
    seed_hex = os.environ.get(env_var)
    if not seed_hex:
        return os.urandom(seed_len)
    master_seed = bytes.fromhex(seed_hex)
    return hashlib.sha256(master_seed + key_idx.to_bytes(4, "little")).digest()
