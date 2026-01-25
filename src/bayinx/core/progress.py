import jax
from tqdm.auto import tqdm

# Registry for progress bars
_pbar_registry: dict[int, tqdm] = {}

def _tqdm_update_host(id_val: int, current_iter, max_iters, message: str):
    # Coerce from JAX arrays as needed
    id_val = int(id_val)
    current_iter = int(current_iter)
    max_iters = int(max_iters)

    # Construct progess bar
    if id_val not in _pbar_registry:
        _pbar_registry[id_val] = tqdm(total=max_iters, desc=message)

    # Update progress bar
    pbar = _pbar_registry[id_val]
    if current_iter > pbar.n:
        pbar.update(current_iter - pbar.n)

def _tqdm_close_host(id_val, final_iter):
     # Coerce from JAX arrays
    id_val = int(id_val)
    final_iter = int(final_iter)

    # Check for identifier in registry
    if id_val in _pbar_registry:
        pbar = _pbar_registry[id_val]

        # Force progress bar to last iteration count
        if final_iter > pbar.n:
            pbar.update(final_iter - pbar.n)

        # Close progress bar and remove from registry
        pbar.close()
        del _pbar_registry[id_val]


def update_progress(id_val, i, max_iters: int, message: str, print_rate: int):
    """
    Update progress bar.

    Parameters:
        id_val: A unique integer identifier for a progress bar.
        i: Current iteration.
        max_iters: Total iterations.
        print_rate: How often to call back to Python.
    """
    # Only send signal to host every `print_rate` steps
    jax.lax.cond(
        i % print_rate == 0,
        lambda: jax.debug.callback(_tqdm_update_host, id_val, i, max_iters, message),
        lambda: None
    )

def close_progress(id_val, final_iter):
    """
    Terminate progress bar.

    Parameter:
        id_val: A unique integer identifier for a progress bar.
        final_iter: Final iteration to update.
    """
    jax.debug.callback(_tqdm_close_host, id_val, final_iter)
