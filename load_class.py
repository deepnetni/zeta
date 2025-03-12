import importlib


def load_class(class_p: str):
    module_p, module_name = class_p.rsplit(".", 1)
    module = importlib.import_module(module_p)
    cls = getattr(module, module_name)
    return cls
