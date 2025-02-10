import pickle

from joblib import Parallel, delayed

from actinvoting.util_time import current_time, elapsed_time


def equivalent_batch(session, ns, n_jobs=1, file_name=None, force_recompute=False):
    """
    Compute the theoretical equivalents for a list of values of n.

    Parameters
    ----------
    session: WorkingSession
        The working session specifying the culture and the parameters of the model.
    ns: list of int
        The list of values of n (number of voters) for which the values of the theoretical equivalent are to be
        computed.
    n_jobs: int
        The number of parallel jobs to run. If n_jobs=1, the computation is done in a single process.
    file_name: str
        The name of the file where the result is saved. If None, the file name is automatically generated.
    force_recompute: bool
        If True, the computation is done even if the file already exists.

    Returns
    -------
    list of float
        The list of theoretical equivalents for the values of n in the input list.
    """
    # Default parameters
    culture = session.culture
    c = session.c
    if file_name is None:
        file_name = (str(culture) + f"_{c=}_{ns=}_equiv").\
                        replace(' ', '_').replace('/', '_') + '.pkl'
    if len(file_name) >= 255:
        file_name = (str(culture) + f"_{c=}_hash(ns)={hash(tuple(ns))}_equiv").\
                        replace(' ', '_').replace('/', '_') + '.pkl'

    # Try to load the file
    if not force_recompute:
        try:
            with open(file_name, 'rb') as f:
                print(f"Loading {file_name}")
                return pickle.load(f)
        except FileNotFoundError:
            pass

    # Define the function to be parallelized
    def proba_equivalent(n):
        return float(session.equivalent(n=n))

    # Run the parallelized function
    start_time = current_time()
    result = list(Parallel(n_jobs=n_jobs)(delayed(proba_equivalent)(n) for n in ns))
    run_time_seconds, run_time_str = elapsed_time(start_time)
    print(f'{run_time_str=}')

    # Create the file if it does not exist, then save the result in the file
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)
    return result
