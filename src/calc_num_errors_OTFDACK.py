import numpy as np
import logging
import sys
import random
import json
from pathlib import Path
from util import load_config, load_lib, rm_decoder_result_wo_noise, setup_logging, calculate_num_errors_first_n


class OTFDACKOracle:
    def __init__(self, lib, scheme, n2, n, n1, w, use_error_patterns, templates,rho):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2
        self.n = n
        self.n1 = n1
        self.w = w
        self.use_error_patterns = use_error_patterns
        self.templates = templates
        self.rho = rho

    def decode_result(self, vector_sum):
        return rm_decoder_result_wo_noise(eXORu=vector_sum, lib=self.lib)

    def generate_full_vector_dropping_excess(self):
        vector = np.zeros(self.n, dtype=int)
        vector[:self.w] = 1
        np.random.shuffle(vector)
        return vector[:self.n1 * self.n2]
    
    def obtain_probabilities(self, vector):
        vector_blocks = np.array_split(vector, int(len(vector)/self.n2))
        vector_probability_blocks = np.empty_like(vector_blocks, dtype=float)

        for i, vector_block in enumerate(vector_blocks):
            block_probability = np.ones(self.n2)
            for j, error_pattern in enumerate(self.use_error_patterns):

                vector_sum = np.bitwise_xor(vector_block, error_pattern)

                mtmp = self.decode_result(vector_sum)
                mtmp = str(mtmp)
                
                # Handle noise and missing keys
                # rho is the oracle error probability
                if np.random.binomial(1, self.rho) != 0:
                    mtmp = 999
                template = self.templates[j]
                if mtmp not in template:
                    mtmp = random.choice(list(template.keys()))
                probability = np.array(template[mtmp])
                block_probability *= probability
            vector_probability_blocks[i] = block_probability
        v = vector_probability_blocks.ravel()
        vector_probability_blocks_xy = np.concatenate([v, v])   
        return vector_probability_blocks_xy
    def sample_and_sort_according_to_probability(self):
        x = self.generate_full_vector_dropping_excess()
        y = self.generate_full_vector_dropping_excess()
        xy = np.concatenate((x, y))
        xXORy = np.bitwise_xor(x, y)
        xy_probabilities = self.obtain_probabilities(vector=xXORy)
        sorted_indices = np.argsort(xy_probabilities)
        sorted_xy = xy[sorted_indices]
        sorted_xy_probabilities = xy_probabilities[sorted_indices]
        num_errors_first_n = calculate_num_errors_first_n(n=self.n,array=sorted_xy)
        return sorted_xy, sorted_xy_probabilities, num_errors_first_n


def load_template(scheme, num_error_patterns):
    template_path = Path(f"./template/{scheme}/error_patterns_templates_{scheme}.json")
    with open(template_path, 'r', encoding='utf-8') as file:
        template_entries = json.load(file)

    if num_error_patterns > len(template_entries):
        raise ValueError(f"Requested {num_error_patterns} error patterns, but only {len(template_entries)} templates are available.")

    use_error_patterns_index = list(range(num_error_patterns))
    logging.info("The error patterns to use has index {}".format(use_error_patterns_index))

    use_error_patterns = []
    templates = []
    for i in use_error_patterns_index:
        use_error_patterns.append(np.array(template_entries[i]['error_pattern'], dtype=int))
        templates.append(template_entries[i]['template'])

    return use_error_patterns, templates


def main(scheme, rho, num_error_patterns):
    # Load configuration and setup
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)

    n2 = params.get("n2", "n2_not_specified")
    n = params.get("n", "n_not_specified")
    n1 = params.get("n1", "n1_not_specified")
    w = params.get("w", "w_not_specified")

    setup_logging(script='calc_num_errors_OTFDACK', scheme=scheme)

    # load error patterns and corresponding templates. 
    use_error_patterns, templates = load_template(scheme, num_error_patterns)
    
    # initialize the oracle
    # rho is the oracle error probability
    oracle = OTFDACKOracle(
        lib=lib,
        scheme=scheme,
        n2=n2,
        n=n,
        n1=n1,
        w=w,
        use_error_patterns=use_error_patterns,
        templates=templates,
        rho=rho
    )

    num_keys = 1000
    num_errors_first_n_list = []
    
    for _ in range(num_keys):
        if _ % 100 == 0:
            logging.info(f'calculating num_errors_first_n for key {_+1}/{num_keys}...')
        sorted_xy, sorted_xy_probabilities, num_errors_first_n = oracle.sample_and_sort_according_to_probability()

        num_errors_first_n_list.append(num_errors_first_n)

    # save  num_errors_first_n_list
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    output_filename = f"ot_num_errors_first_n_{scheme}_rho={rho}_num_error_patterns={num_error_patterns}_num_keys={num_keys}.npy"
    np.save(output_dir/output_filename, np.array(num_errors_first_n_list))
    logging.info(f"scheme={scheme}, rho={rho}, num_keys={num_keys}, average num_errors_first_n={np.mean(num_errors_first_n_list)}. Results saved to {output_filename}.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme> <rho> <num_error_patterns> ")
        sys.exit(1)
    main(scheme=sys.argv[1], rho=float(sys.argv[2]), num_error_patterns=int(sys.argv[3]))
