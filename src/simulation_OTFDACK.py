import numpy as np
import logging
import sys
import random
import json
from pathlib import Path
from isd import ISDSimulator
from util import load_config, load_lib, rm_decoder_result_wo_noise, setup_logging


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
        return sorted_xy, sorted_xy_probabilities


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
    l = params.get("l", "l_not_specified")

    setup_logging(script='simulationOT', scheme=scheme)

    # load error patterns and corresponding templates. 
    use_error_patterns, templates = load_template(scheme, num_error_patterns)
    
    # initialize the simulators
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
    # experiment with this parameter 
    select_scale = 1.2
    isdsim = ISDSimulator(n=n, l=l, select_scale=select_scale)

    # ==========================================================
    # CONFIGURATION: Choose which statistic to calculate
    # Options: "p_one_shot" or "exp_num_draws"
    STATISTIC_TO_CALCULATE = "exp_num_draws" 
    # ==========================================================

    num_keys = 1000

    if STATISTIC_TO_CALCULATE == "p_one_shot":
        calculation_function = isdsim.calculate_p_one_shot_for_one_key
        num_exp = 100000 # Number of experiments for each key to estimate p_one_shot. Slow.
        func_kwargs = {'num_experiments': num_exp} # Additional parameters 
        logging.info(f"Calculated p_one_shot for {num_keys} random keys.")
        output_filename = f"ot_p_one_shot_{scheme}_rho={rho}_select_scale={select_scale}_num_error_patterns={num_error_patterns}_num_keys={num_keys}_num_experiments={num_exp}.npy"
    elif STATISTIC_TO_CALCULATE == "exp_num_draws":
        calculation_function = isdsim.experiment_num_draws_for_one_key
        func_kwargs = {} # No extra arguments for this function
        logging.info(f"Experiment number of draws to succeed for {num_keys} random keys (bounded by T=1024).")
        output_filename = f"ot_exp_num_draws_{scheme}_rho={rho}_select_scale={select_scale}_num_error_patterns={num_error_patterns}_num_keys={num_keys}.npy"
    else:
        raise ValueError(f"Unknown statistic type: '{STATISTIC_TO_CALCULATE}'")
    
    
    all_results = []
    
    for _ in range(num_keys):
        if _ % 100 == 0:
            logging.info(f'calculating {STATISTIC_TO_CALCULATE} for key {_+1}/{num_keys}...')
        sorted_xy, sorted_xy_probabilities = oracle.sample_and_sort_according_to_probability()

        result = calculation_function(
            sorted_xy=sorted_xy, 
            sorted_xy_probabilities=sorted_xy_probabilities, 
            **func_kwargs
        )
        
        all_results.append(result)

    # save results
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir/output_filename, np.array(all_results))
    logging.info(f"Results saved to {output_filename}.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme> <rho> <num_error_patterns> ")
        sys.exit(1)
    main(scheme=sys.argv[1], rho=float(sys.argv[2]), num_error_patterns=int(sys.argv[3]))
