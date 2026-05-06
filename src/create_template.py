import numpy as np
from collections import defaultdict
import sys
import json
import pandas as pd
import os
import re
from util import sample_vector_from_scheme, load_config, load_lib, setup_logging, rm_decoder_result_wo_noise
import logging


class TemplateCreator:
    def __init__(self, lib, scheme, n2, config):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2
        self.config = config
    
    def select_error_patterns(self, error_patterns):

        # Use output directly from find_error_patterns.py
        # Filter and sort error patterns based on entropy threshold
        entropy_threshold = self.config['entropy_threshold']
        selected = error_patterns[error_patterns['Entropy'] >= entropy_threshold].sort_values(by='Entropy', ascending=False)
        if selected.empty:
            raise SystemExit(f"No error patterns with entropy >= {entropy_threshold}.")
        selected_error_patterns = np.stack(selected['Vectors'].to_numpy())
        return selected_error_patterns

    def append_entry_to_json(self, file_path, entry):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                entries = json.load(file)
        else:
            entries = []

        entries.append(entry)
        with open(file_path, 'w', encoding='utf-8') as file:
            self.dump_entries_with_compact_vectors(entries, file)

    def dump_entries_with_compact_vectors(self, entries, file):
        for entry in entries:
            if 'template' in entry:
                entry['template'] = dict(sorted(entry['template'].items(), key=lambda item: int(item[0])))
            if 'percentages' in entry:
                entry['percentages'] = dict(sorted(entry['percentages'].items(), key=lambda item: int(item[0])))

        json_str = json.dumps(entries, indent=4)

        def compact_list(match):
            compacted = re.sub(r'\s+', ' ', match.group(1)).strip()
            return f"[{compacted}]"

        json_str = re.sub(r'\[\s*([\d\.\,\s\-eE]+)\s*\]', compact_list, json_str)
        file.write(json_str)

    def create_template(self, error_patterns):
        selected_patterns = self.select_error_patterns(error_patterns)
        num_samples = self.config['num_samples']
  
        for i in range(len(selected_patterns)): 
            error_pattern = np.array(selected_patterns[i])
            results_distribution = defaultdict(int)
            u_aggregate = defaultdict(lambda: np.zeros(self.n2, dtype=int))

            for _ in range(num_samples):
                u1 = sample_vector_from_scheme(self.scheme)
                u2= sample_vector_from_scheme(self.scheme)
                u = np.bitwise_xor(u1, u2)
                vector_sum = np.bitwise_xor(u, error_pattern)
                rm_decoder_result = rm_decoder_result_wo_noise(eXORu=vector_sum, lib=self.lib)
                u_aggregate[rm_decoder_result] += u
                results_distribution[rm_decoder_result] += 1

            # Calculate probability template
            probability_template = {}
            decoder_result_percentages = {}
            total_samples = sum(results_distribution.values())

            for decoder_result in list(u_aggregate.keys()): # Iterate over a copy of keys
                decoder_result_percentage = (results_distribution[decoder_result] / total_samples) * 100
                logging.info(f'Error pattern {i}: decoder result {decoder_result}, percentage {decoder_result_percentage:.4f}%.')
                
                filter_threshold = (1 / 2000) * 100
                
                if decoder_result_percentage >= filter_threshold:
                    # This result passes the threshold
                    decoder_result_key = str(decoder_result)
                    probability_template[decoder_result_key] = (
                        u_aggregate[decoder_result] / results_distribution[decoder_result]
                    ).tolist()
                    decoder_result_percentages[decoder_result_key] = decoder_result_percentage
                else:
                    # This result does not pass, remove it
                    del u_aggregate[decoder_result]
                    del results_distribution[decoder_result]

            error_pattern_list = error_pattern.astype(int).reshape(-1).tolist()
            template_entry = {
                'error_pattern': error_pattern_list,
                'template': probability_template,
            }
            percentage_entry = {
                'error_pattern': error_pattern_list,
                'percentages': decoder_result_percentages,
            }
            
            template_path = self.config['template_path']
            os.makedirs(template_path, exist_ok=True)
            templates_file_path = os.path.join(template_path, f'error_patterns_templates_{self.scheme}.json')
            percentages_file_path = os.path.join(template_path, f'error_patterns_percentages_{self.scheme}.json')
        
            self.append_entry_to_json(templates_file_path, template_entry)
            self.append_entry_to_json(percentages_file_path, percentage_entry)
            logging.info(f"Saved template and percentages for error pattern {i}")

def main(scheme):
    # Load configuration and setup
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)

    n2 = params.get("n2", "n2_not_specified") # Consider raising error if not found or using actual default

    # Setup logging
    setup_logging(script="create_template", scheme=scheme) 

    # load the optimized error patterns
    error_patterns = pd.read_csv(f"./output/optimized_error_patterns_{scheme}.csv")
    # Parse the Vectors column into NumPy arrays
    error_patterns['Vectors'] = error_patterns['Vectors'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ").astype(int) if isinstance(x, str) else np.array(x, dtype=int))

    # Configuration 
    config = {
        'entropy_threshold': 4.15, # give a threshold. Templates will be created for optimized error patterns with entropies above this threshold.
        'num_samples': 100000000, 
        'template_path': f"./template/{scheme}/"
    }

    # Call the template creator
    template_creator = TemplateCreator(
        lib=lib,
        scheme=scheme,
        config=config,
        n2=n2
    )

    template_creator.create_template(error_patterns)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <scheme>")
        sys.exit(1)
    
    main(scheme=sys.argv[1])
