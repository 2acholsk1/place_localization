from typing import Optional, Dict, Any

from joblib import Parallel
from tqdm.auto import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm: bool = True, total: Optional[int] = None,
                 tqdm_options: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_tqdm = use_tqdm
        self._total = total
        self._tqdm_options = tqdm_options or {}

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, **self._tqdm_options) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()