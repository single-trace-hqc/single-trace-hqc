import numpy as np
import math
import random

class ISDSimulator:
    """
    A simulator to calculate the success rate of an ISD attack 
    based on soft information (probabilities) of the secret key bits.
    """
    def __init__(self, n, l, select_scale=1.1):
        """
        Initializes the ISDSimulator with the necessary parameters.

        Args:
            n (int): The length of the code.
            l (int): The number of extra positions to consider.
            select_scale (float): A factor to determine the range of bits to select.
        """
        self.n = n
        self.l = l
        self.select_range = math.ceil(select_scale * self.n / 4) * 4
        self.maxW = 6 # maxW/2: max list weight for collision

    def _split_array(self, sorted_array, sorted_array_probabilities):
        """
        Splits the input array into two lists based on a specific pairing and random selection strategy.
        """
        
        array = sorted_array[:self.select_range]
        array_probabilities = sorted_array_probabilities[:self.select_range]
        
        if len(array) % 4 != 0:
            raise ValueError("The length of the array to be split (select_range) must be a multiple of 4.")

        array1, array2 = [], []
        array1_probabilities, array2_probabilities = [], []

        for i in range(0, len(array), 4):
            if random.choice([True, False]):
                array1.extend([array[i], array[i+3]])
                array1_probabilities.extend([array_probabilities[i], array_probabilities[i+3]])
                array2.extend([array[i+1], array[i+2]])
                array2_probabilities.extend([array_probabilities[i+1], array_probabilities[i+2]])
            else:
                array1.extend([array[i+1], array[i+2]])
                array1_probabilities.extend([array_probabilities[i+1], array_probabilities[i+2]])
                array2.extend([array[i], array[i+3]])
                array2_probabilities.extend([array_probabilities[i], array_probabilities[i+3]])
        
        array1 = np.array(array1)
        array2 = np.array(array2)
        array1_probabilities = np.array(array1_probabilities)
        array2_probabilities = np.array(array2_probabilities)

        ##### weighted sampling from array1 and array2
        # **weight function**: 1/probability of being 1.
        array1_weights = 1/array1_probabilities
        array2_weights = 1/array2_probabilities
        # Normalize weights
        array1_weights /= np.sum(array1_weights)
        array2_weights /= np.sum(array2_weights)

        return array1, array2, array1_weights, array2_weights

    def _weighted_sampling_and_return_num_draws(self, array1, array2, array1_weights, array2_weights, T):
        """
        Weighted sampling and returns the number of draws to succeed.
        """
        list_size = (self.n + self.l) // 2
        num_draw = float('inf')

        for j in range(T):
            list1 = np.random.choice(array1, size=list_size, replace=False, p=array1_weights)
            list2 = np.random.choice(array2, size=list_size, replace=False, p=array2_weights)

            if (list1.sum() + list2.sum()) > self.maxW:
                continue

            if list1.sum() <= self.maxW / 2 and list2.sum() <= self.maxW / 2:
                num_draw = j + 1
                break
        
        return num_draw

    def calculate_p_one_shot_for_one_key(self, sorted_xy, sorted_xy_probabilities, num_experiments=100000):
        """
        Calculates the one-shot success probability for a given key and probabilities.

        Args:
            sorted_xy: The array of bits, sorted by probability.
            sorted_xy_probabilities: The probabilities associated with sorted_xy.
            num_experiments (int): The number of simulation runs to average over.

        Returns:
            The calculated one-shot success probability.
        """
        array1, array2, array1_weights, array2_weights = self._split_array(
            sorted_xy, sorted_xy_probabilities
        )

        num_draws_list = []
        for _ in range(num_experiments):
            num_draws = self._weighted_sampling_and_return_num_draws(
                array1, array2, array1_weights, array2_weights, T=1
            )
            num_draws_list.append(num_draws)

        p_one_shot = sum(1 for x in num_draws_list if math.isfinite(x)) / num_experiments
        return p_one_shot
    
    def experiment_num_draws_for_one_key(self, sorted_xy, sorted_xy_probabilities):
        """
        Records the number of draws to succeed for one key, bounded by T trials.

        If a feasible draw is found within T trials, it returns the number of draws.
        Otherwise, it returns infinity.

        Args:
            sorted_xy: The array of bits, sorted by probability.
            sorted_xy_probabilities: The probabilities associated with sorted_xy.

        Returns:
            The number of draws to succeed, or float('inf') if unsuccessful.
        """
        
        array1, array2, array1_weights, array2_weights = self._split_array(
            sorted_xy, sorted_xy_probabilities
        )
 
        num_draws = self._weighted_sampling_and_return_num_draws(
            array1, array2, array1_weights, array2_weights, T=1024
        )

        return num_draws


