from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_spawned_objectnav_dataset():
    try:
        from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
                import SpawnedObjectNavDatasetV0
    except ImportError as e:
        spawned_objectnav_import_error = e

        @registry.register_dataset(name="SpawnedObjectNav-v0")
        class SpawnedObjectNavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise spawned_objectnav_import_error


