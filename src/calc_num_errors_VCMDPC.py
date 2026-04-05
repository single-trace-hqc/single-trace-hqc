import sys
import logging
import ctypes
import numpy as np
from scipy.sparse import identity, hstack
from pathlib import Path
import gc
from util import setup_logging, load_config, generate_random_binary_vector, apply_bit_flip, sparse_circulant, calculate_num_errors_first_n

def calculate_initial_llr(noisy_obs, n, wy, rho):
    """Calculates the initial LLR vectors for syndrome bits and variable nodes."""
    epsilon = 1e-12
    rho_clamped = np.clip(rho, epsilon, 1 - epsilon)
    log_term = np.log((1 - rho_clamped) / rho_clamped)
    
    syndrome_llr = np.where(noisy_obs == 1, -log_term, log_term)

    xy_llr_val = np.log((n - wy) / wy)
    xy_llr = np.full(2 * n, xy_llr_val, dtype=np.double)
    
    return np.concatenate((xy_llr, syndrome_llr)).astype(np.double)


def prepare_sparse_matrix_for_c(H_sparse, n, n1,n2):
    """Takes a sparse H, creates P=[H|I] and extracts CSR/CSC components for C."""
    I_sparse = identity(n, format='csr', dtype=np.int32)
    P_sparse_csr = hstack([H_sparse, I_sparse], format='csr')

    # Truncate to keep only the first n1 * n2 rows
    num_rows_to_keep = n1 * n2
    P_sparse_csr = P_sparse_csr[:num_rows_to_keep, :]
    
    # Get the CSC format
    P_sparse_csc = P_sparse_csr.tocsc()

    # Extract necessary parameters
    m, n_c = P_sparse_csr.shape
    nnz = P_sparse_csr.nnz
    
    matrix_params = {
        'm': m, 
        'n_c': n_c, 
        'nnz': nnz,
        'row_offsets_csr': P_sparse_csr.indptr.astype(np.int32),
        'col_indices_csr': P_sparse_csr.indices.astype(np.int32),
        'row_degrees': P_sparse_csr.getnnz(axis=1).astype(np.int32),
        'col_offsets_csc': P_sparse_csc.indptr.astype(np.int32),
        'row_indices_csc': P_sparse_csc.indices.astype(np.int32),
        'col_degrees': P_sparse_csc.getnnz(axis=0).astype(np.int32)
    }
    return matrix_params


def setup_mdpcbp(lib_path):
    """Loads the C library, sets the expected argument and return types, and returns the C function."""
    try:
        mdpcbp_lib = ctypes.CDLL(lib_path)
        mdpc_decode_func = mdpcbp_lib.mdpc_decode
        mdpc_decode_func.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, # m, n_c, num_s, max_iter, nnz
            ctypes.POINTER(ctypes.c_int), # row_offsets (CSR)
            ctypes.POINTER(ctypes.c_int), # col_indices (CSR)
            ctypes.POINTER(ctypes.c_int), # col_offsets (CSC)
            ctypes.POINTER(ctypes.c_int), # row_indices (CSC)
            ctypes.POINTER(ctypes.c_int), # row_degrees
            ctypes.POINTER(ctypes.c_int), # col_degrees
            ctypes.POINTER(ctypes.c_double), # init_llr
            ctypes.POINTER(ctypes.c_double)  # final_llr
        ]
        mdpc_decode_func.restype = None
        return mdpc_decode_func
    except (OSError, AttributeError) as e:
        logging.error(f"Failed to load or configure C library from {lib_path}: {e}")
        sys.exit(1)



def run_mdpc_decode(mdpc_decode_func, matrix_params, init_llr, n, max_iter):
    """Runs the mdpc decoding and returns the final llr."""
    m, n_c, nnz = matrix_params['m'], matrix_params['n_c'], matrix_params['nnz']
    num_s = 2 * n

    row_offsets_ptr = matrix_params['row_offsets_csr'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    col_indices_ptr = matrix_params['col_indices_csr'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    col_offsets_ptr = matrix_params['col_offsets_csc'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    row_indices_ptr = matrix_params['row_indices_csc'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    row_degrees_ptr = matrix_params['row_degrees'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    col_degrees_ptr = matrix_params['col_degrees'].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    init_llr_ptr = init_llr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    final_llr_np = np.zeros(num_s, dtype=np.double)
    final_llr_ptr = final_llr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    try:
        mdpc_decode_func(m, n_c, num_s, max_iter, nnz,
                         row_offsets_ptr, col_indices_ptr,
                         col_offsets_ptr, row_indices_ptr,
                         row_degrees_ptr, col_degrees_ptr,
                         init_llr_ptr, final_llr_ptr)
    except Exception as e:
        logging.error(f"Error during C function call for max_iter={max_iter}: {e}")
        return None, None

    return final_llr_np


def main(scheme, rho, max_iter):
    # Load configuration and setup
    params = load_config(scheme)
    n2 = params.get("n2", "n2_not_specified")
    n = params.get("n", "n_not_specified")
    n1 = params.get("n1", "n1_not_specified")
    w = params.get("w", "w_not_specified") #wy
    l = params.get("l", "l_not_specified")
    wr = params.get("wr", "wr_not_specified")

    setup_logging(script='simulationVC', scheme=scheme)

    # load mdpc bp function from C library
    lib_path = "./src/libMDPCBP.so"
    mdpc_decode_func = setup_mdpcbp(lib_path)
    
    num_keys = 100
    num_errors_first_n_list = []

    for _ in range(num_keys):
        r1 = generate_random_binary_vector(n, wr)
        r2 = generate_random_binary_vector(n, wr)
        H0_sparse = sparse_circulant(r2)
        H1_sparse = sparse_circulant(r1)
        H_sparse = hstack([H0_sparse, H1_sparse], format='csr')
        
        x = generate_random_binary_vector(n, w)
        y = generate_random_binary_vector(n, w)
        xy = np.concatenate((x, y))

        syndrome = H_sparse @ xy % 2
        noisy_syndrome = apply_bit_flip(syndrome, rho)
        initial_llr = calculate_initial_llr(noisy_obs=noisy_syndrome, n=n, wy=w, rho=rho)

        matrix_params = prepare_sparse_matrix_for_c(H_sparse, n, n1, n2)

        
        final_llr = run_mdpc_decode(mdpc_decode_func, matrix_params, initial_llr,  n,max_iter)

        sorted_indices = np.argsort(final_llr)[::-1]
        sorted_xy = xy[sorted_indices]
        sorted_llr = final_llr[sorted_indices]

        num_errors_first_n = calculate_num_errors_first_n(n=n,array=sorted_xy)
        
        num_errors_first_n_list.append(num_errors_first_n)
        
        del H0_sparse, H1_sparse, H_sparse, x, y, xy, syndrome, noisy_syndrome, initial_llr, matrix_params
        gc.collect()

    logging.info(f"Iter = {max_iter}, rho={rho}: average number of errors in the first n positions over {num_keys} keys: {np.mean(num_errors_first_n_list)}")

    # save results and num_errors_at_n_list
    output_dir = Path('./output')
    output_dir.mkdir(exist_ok=True)
    output_name = f"vc_num_errors_first_n_{scheme}_rho={rho}_max_iter={max_iter}_num_keys={num_keys}.npy"
    np.save(output_dir / output_name, np.array(num_errors_first_n_list))
    logging.info(f"Num errors in the first n positions saved to {output_dir / output_name}.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme> <rho> <max_iter> ")
        sys.exit(1)
    main(scheme=sys.argv[1], rho=float(sys.argv[2]), max_iter=int(sys.argv[3]))
