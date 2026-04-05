#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Clamp a value to be within (-0.999999, 0.999999) to avoid issues with atanh
static double clamp(double x) {
    const double limit = 0.999999;
    if (x > limit) return limit;
    if (x < -limit) return -limit;
    return x;
}

// The decoder receives the parity-check matrix as sparse CSR/CSC components:
// CSR: row_offsets and col_indices for row-wise traversing of check nodes.
// CSC: col_offsets and row_indices for column-wise traversing of variable nodes.
// row_degrees, col_degrees, and nnz give the row weights, column weights, and total number of nonzero entries (edges in the Tanner graph).

// MDPC decoding using the sum-product algorithm 
EXPORT void mdpc_decode(int m, int n_c, int num_s, int max_iter, int nnz,
                         const int *row_offsets, const int *col_indices,
                         const int *col_offsets, const int *row_indices,
                         const int *row_degrees, const int *col_degrees,
                         const double *init_llr, double *final_llr) {

    // Allocate message arrays (indexed according to CSR order)
    double *v2c = (double*)malloc(nnz * sizeof(double)); // Message from variable to check
    double *c2v = (double*)malloc(nnz * sizeof(double)); // Message from check to variable
    double *beliefs = (double*)malloc(n_c * sizeof(double)); // Belief of variable nodes (n_c = number of columns)

    if (!v2c || !c2v || !beliefs) {
        fprintf(stderr, "Memory allocation failed for message arrays\n");
        free(v2c);
        free(c2v);
        free(beliefs);
        return;
    }

    // Initialize messages (v2c) and beliefs based on initial LLRs
    // Initialize v2c based on CSR order
    for (int j = 0; j < m; ++j) { // Iterate through check nodes (rows)
        for (int csr_k = row_offsets[j]; csr_k < row_offsets[j+1]; ++csr_k) {
            int var_node_idx = col_indices[csr_k]; // Get connected variable node
            // Initialize message from variable var_node_idx to check node j
            v2c[csr_k] = init_llr[var_node_idx]; // v2c indexed by CSR order
            c2v[csr_k] = 0.0; // Initialize c2v messages to 0
        }
    }
    // Initialize beliefs for all variable nodes
    for(int i=0; i<n_c; ++i) { // n_c is the total number of variable nodes
        beliefs[i] = init_llr[i];
    }


    // Iterative decoding
    for (int iter = 0; iter < max_iter; iter++) {

        // --- Check Node Update ---
        // Uses CSR structure: row_offsets, col_indices
        // Reads v2c[CSR index], Writes c2v[CSR index]
        for (int j = 0; j < m; j++) { // Iterate through check nodes (rows)
            int current_row_degree = row_degrees[j];
            assert(current_row_degree > 1);

            // Allocate temporary storage for tanh values for the current check node
            double *tanh_half_v2c = (double*)malloc(current_row_degree * sizeof(double));
            if (!tanh_half_v2c) {
                fprintf(stderr, "Memory allocation failed for tanh_half_v2c\n");
                goto cleanup;
            }

            // Compute tanh(v2c/2) for all incoming messages and the product of these terms
            double total_prod_tanh = 1.0;
            for (int k = 0; k < current_row_degree; k++) {
                int csr_k = row_offsets[j] + k; // CSR index for the edge
                double tanh_val = tanh(v2c[csr_k] / 2.0);
                tanh_half_v2c[k] = tanh_val;
                total_prod_tanh *= tanh_val;
            }

            // Compute outgoing c2v messages
            for (int k = 0; k < current_row_degree; k++) {
                int csr_k = row_offsets[j] + k;
                double incoming_tanh = tanh_half_v2c[k];

                // Calculate product excluding the current input incoming_tanh
                double prod_excluding_k;
                if (fabs(incoming_tanh) < 1e-20) {
                    prod_excluding_k = 1.0;
                    for (int l = 0; l < current_row_degree; ++l) {
                        if (k != l) {
                            prod_excluding_k *= tanh_half_v2c[l];
                        }
                    }
                } else {
                    prod_excluding_k = total_prod_tanh / incoming_tanh;
                }

                // Clamp the value before atanh to avoid numerical issues
                double clamped_prod = clamp(prod_excluding_k);

                // Calculate the outgoing message c2v = 2 * atanh
                c2v[csr_k] = 2.0 * atanh(clamped_prod);
            }

            free(tanh_half_v2c);
        }


        // --- Variable Node Update ---
        // Uses CSC structure: col_offsets, row_indices
        // Reads c2v[CSR index], Writes v2c[CSR index] and updates beliefs[i]
        for (int i = 0; i < n_c; i++) { // Iterate through variable nodes (columns)
            double sum_c2v_messages = 0;
            int current_col_degree = col_degrees[i];
            if (current_col_degree == 0) {
                 beliefs[i] = init_llr[i];
                 continue;
            }

            // Sum incoming c2v messages for variable node i
            // Iterate through edges connected to variable i using CSC structure
            for (int k = col_offsets[i]; k < col_offsets[i + 1]; k++) {

                int j = row_indices[k]; // The check node connected by this edge
                // Mapping from CSC edge index k to CSR edge index csr_k
                int csr_k = -1;
                for (int temp_csr_k = row_offsets[j]; temp_csr_k < row_offsets[j+1]; ++temp_csr_k) {
                    if (col_indices[temp_csr_k] == i) {
                        csr_k = temp_csr_k;
                        break;
                    }
                }
                
                // Summing up all incoming c2v messages for variable node i
                assert(csr_k != -1);
                sum_c2v_messages += c2v[csr_k];
            }

            // Update belief for variable node i
            // Normalize belief by (degree + 1)
            beliefs[i] = (init_llr[i] + sum_c2v_messages) / (double)(current_col_degree + 1);

            // Compute outgoing v2c messages from variable node i
            for (int k = col_offsets[i]; k < col_offsets[i + 1]; k++) {

                int j = row_indices[k];

                // Again, mapping from CSC edge index k to CSR edge index csr_k
                int csr_k = -1;
                for (int temp_csr_k = row_offsets[j]; temp_csr_k < row_offsets[j+1]; ++temp_csr_k) {
                    if (col_indices[temp_csr_k] == i) {
                        csr_k = temp_csr_k;
                        break;
                    }
                }

                // Calculate sum of initial LLR and other incoming c2v messages
                v2c[csr_k] = (init_llr[i] + sum_c2v_messages - c2v[csr_k]);
            }
        }
    }

    // Compute final LLRs for the specified variables (first num_s variable nodes)
    for (int i = 0; i < num_s; i++) {
        final_llr[i] = beliefs[i];
    }

cleanup:
    free(v2c);
    free(c2v);
    free(beliefs);
}
