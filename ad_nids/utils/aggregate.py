
import multiprocessing as mp
from tqdm import tqdm


def aggregate_features_pool(grouped, extract_features_wkr, processes=1):

    pbar = tqdm(total=len(grouped))

    records = []

    def log_progress(result):
        records.append(result)
        pbar.update(1)

    if processes > 1:
        pool = mp.Pool(processes=processes)
        for name, grp in grouped:
            pool.apply_async(
                extract_features_wkr, ((name, grp),),
                callback=log_progress)
        pool.close()
        pool.join()
    else:
        for name, grp in grouped:
            res = extract_features_wkr((name, grp))
            log_progress(res)

    return records
