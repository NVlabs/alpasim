"""
This script is heavily based on the EasyDict class from `easydict` package.
"""
import logging

logger = logging.getLogger(__name__)

class AttrDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        
        # first check if value is a nested dict
        # This change is tailored for torch models' `state_dict`.    
        updated_dict = {}
        for k, v in d.items():
            if "." in k:
                k1, k2 = k.split(".", 1)
                if k1 not in updated_dict:
                    updated_dict[k1] = {}
                updated_dict[k1][k2] = v
            elif k in updated_dict:
                assert isinstance(v, dict), f"Conflicting value for key `{k}` found. Value should be a dict."
                for subkey in v:
                    if subkey in updated_dict[k]:
                        assert updated_dict[k][subkey] == v[subkey], f"Conflicting value for key `{k}.{subkey}` found."
                        logger.warning(f"Duplicate key `{k}.{subkey} found.")
                updated_dict[k].update(v)
            else:
                updated_dict[k] = v
        
        for k, v in updated_dict.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(AttrDict, self).__setattr__(name, value)
        super(AttrDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(AttrDict, self).pop(k, d)


def convert_to_attribute_dict(obj: dict) -> AttrDict:
    attributes = AttrDict(obj)
    return attributes