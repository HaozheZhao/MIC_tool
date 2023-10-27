import importlib
import os

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def fetch_task_id(dir):
    all_tasks = os.listdir(dir)
    if not all_tasks:
        task_id = "%05d" % (1)
    else:
        task_id = "%05d" % (max([int(i.split('-')[0].strip('task')) for i in all_tasks])+1)
    return task_id