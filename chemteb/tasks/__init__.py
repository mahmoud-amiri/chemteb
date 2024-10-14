from __future__ import annotations
import pkgutil
import importlib
import inspect
from mteb.tasks import AbsTask


package = importlib.import_module(__name__)

CHEMICAL_TASKS = []

for loader, module_name, is_pkg in pkgutil.walk_packages(
    package.__path__, package.__name__ + "."
):
    if not is_pkg:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AbsTask) and not obj.__module__.startswith("mteb"):
                CHEMICAL_TASKS.append(obj)