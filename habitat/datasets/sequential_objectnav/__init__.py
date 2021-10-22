from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_sequential_objectnav_dataset():
    try:
        from habitat.datasets.sequential_objectnav.sequential_objectnav_dataset \
                import SequentialObjectNavDatasetV0
    except ImportError as e:
        sequential_objectnav_import_error = e

        @registry.register_dataset(name="SequentialObjectNav-v0")
        class SequentialObjectNavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise sequential_objectnav_import_error


