from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_sequential_nav_task():
    try:
        from habitat.tasks.sequential_nav.sequential_nav import SequentialNavigationTask  # noqa
    except ImportError as e:
        seq_nav_task_import_error = e

        @registry.register_task(name="SequentialNav-v0")
        class SequentialNavTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise seq_nav_task_import_error


def _try_register_sequential_objectnav_task():
    try:
        from habitat.tasks.sequential_nav.sequential_objectnav import SequentialObjectNavTask  # noqa
    except ImportError as e:
        seq_objectnav_task_import_error = e

        @registry.register_task(name="SequentialObjectNav-v0")
        class SequentialObjectNavTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise seq_objectnav_task_import_error

