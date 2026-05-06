import numpy as np
from collections import defaultdict
import copy
import logging
import sys
import os
import csv
from filelock import FileLock
from util import sample_binary_vector_in_weight_range, sample_vector_from_scheme, rm_decoder_result_wo_noise, mutate2bits, mutate_bit_by_bit, load_config, load_lib, setup_logging


class ErrorPatternFinder:
    def __init__(self, lib, scheme, n2):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2

    def calculate_shannon_entropy(self, results_distribution):
        frequencies = np.array(list(results_distribution.values()))
        total_samples = frequencies.sum()
        probabilities = frequencies / total_samples
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def evaluate_error_pattern(self, error_pattern, num_iter):
        results_distribution = defaultdict(int)
        for _ in range(num_iter):
            u1 = sample_vector_from_scheme(self.scheme)
            u2 = sample_vector_from_scheme(self.scheme)
            u = np.bitwise_xor(u1, u2)
            vector_sum = np.bitwise_xor(u, error_pattern)
            rm_decoder_result = rm_decoder_result_wo_noise(eXORu=vector_sum, lib=self.lib)

            results_distribution[rm_decoder_result] += 1
        shannon_entropy = self.calculate_shannon_entropy(results_distribution)
        return shannon_entropy, results_distribution

    def write_results_to_csv(self, filepath, vectors, entropy):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        lockfile = filepath + '.lock'
        file_exists = os.path.exists(filepath)

        with FileLock(lockfile):
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                # If file is new, write headers
                if not file_exists:
                    writer.writerow(['Vectors', 'Entropy'])
                writer.writerow([vectors, entropy])

    def find_optimal_error_pattern(self, num_generation, num_children, output_filepath):
        if self.scheme == 'hqc1':
            e_0 = sample_binary_vector_in_weight_range(self.n2, 0.42, 0.44)
        else:
            e_0 = sample_binary_vector_in_weight_range(self.n2, 0.435, 0.460)

        shannon_entropy_0, results_distribution_0 = self.evaluate_error_pattern(error_pattern=e_0, num_iter=20000)
        logging.info("Starting vector Shannon entropy: {}".format(shannon_entropy_0))

        generation_best_shannon_entropy = shannon_entropy_0
        generation_best_vectors = copy.deepcopy(e_0)
        generation_best_results_distribution = results_distribution_0

        for generation in range(num_generation):
            children_best_shannon_entropy = generation_best_shannon_entropy
            children_best_vectors = copy.deepcopy(generation_best_vectors)
            children_best_results_distribution = generation_best_results_distribution

            for _ in range(num_children):
                if generation_best_shannon_entropy < 3.6:
                    new_mutation = mutate2bits(copy.deepcopy(generation_best_vectors))
                    num_iter = 20000
                else:
                    new_mutation = mutate_bit_by_bit(copy.deepcopy(generation_best_vectors))
                    num_iter = 50000

                new_mutation_shannon_entropy, new_mutation_results_distribution = self.evaluate_error_pattern(error_pattern=new_mutation, num_iter=num_iter)

                if new_mutation_shannon_entropy > children_best_shannon_entropy:
                    children_best_shannon_entropy = new_mutation_shannon_entropy
                    children_best_vectors = new_mutation
                    children_best_results_distribution = new_mutation_results_distribution

            if children_best_shannon_entropy > generation_best_shannon_entropy:
                generation_best_vectors = children_best_vectors
                generation_best_shannon_entropy = children_best_shannon_entropy
                generation_best_results_distribution = children_best_results_distribution
                logging.info("generation {} best Shannon entropy: {}".format(generation, generation_best_shannon_entropy))
                logging.info("Current best results distribution: {}. Number of keys: {}".format(generation_best_results_distribution, len(generation_best_results_distribution)))

        
        logging.info("Optimized vector Shannon entropy: {}".format(generation_best_shannon_entropy))
        self.write_results_to_csv(output_filepath, generation_best_vectors, generation_best_shannon_entropy)



def main(scheme):
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)

    n2 = params.get("n2", "n2_not_specified")
    
    # Setup logging
    setup_logging(script="find_error_patterns", scheme=scheme)
    
    # Create an instance of ErrorPatternFinder
    error_finder = ErrorPatternFinder(lib=lib, scheme=scheme, n2=n2)
    
    # Start finding the optimal error pattern
    os.makedirs('output', exist_ok=True)
    output_filepath = 'output/optimized_error_patterns_{}.csv'.format(scheme)
    error_finder.find_optimal_error_pattern(num_generation=120, num_children=50, output_filepath=output_filepath)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <scheme>")
        sys.exit(1)
    
    main(scheme=sys.argv[1])
