"""
This code is adapted from, https://github.com/SiggiGue/hdfdict, which is licensed under MIT permissions.
"""

# %%
from collections import UserDict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import yaml
from numpy import str_

TYPE = '_type_'

@contextmanager
def hdf_file(hdf, lazy=True, *args, **kwargs):
    """Context manager that yields an h5 file if `hdf` is a string,
    otherwise it yields hdf as is."""
    if isinstance(hdf, (str, Path)):
        if not lazy:  
            # The file can be closed after reading
            # therefore the context manager is used.
            with h5py.File(hdf, *args, **kwargs) as hdf:
                yield hdf
        else:
            # The file should stay open because datasets 
            # are read on item access.
            yield h5py.File(hdf, *args, **kwargs)
    else:
        yield hdf


def unpack_dataset(item):
    """Reconstruct a hdfdict dataset.
    Only some special unpacking for yaml and datetime types.

    Parameters
    ----------
    item : h5py.Dataset

    Returns
    -------
    key: Unpacked key
    value : Unpacked Data
    
    """
    value = item[()]
    type_id = item.attrs.get(TYPE, str_()).astype(str)
    if type_id == 'datetime':
        if hasattr(value, '__iter__'):
            value = [datetime.fromtimestamp(
                ts) for ts in value]
        else:
            value = datetime.fromtimestamp(value)

    elif type_id == 'yaml':
        value = yaml.safe_load(value.decode())
    
    elif type_id == 'list':
        value = list(value)

    elif type_id == 'tuple':
        value = tuple(value)

    elif type_id == 'str':
        value = str_(value).astype(str)
    
    return value


class LazyHdfDict(UserDict):
    """Helps loading data only if values from the dict are requested.

    This is done by reimplementing the __getitem__ method.

    """
    def __init__(self, _h5file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._h5file = _h5file  # used to close the file on deletion.

    def __getitem__(self, key):
        """Returns item and loads dataset if needed."""
        item = super().__getitem__(key)
        if isinstance(item, h5py.Dataset):
            item = unpack_dataset(item)
            self.__setitem__(key, item)
        return item

    def unlazy(self):
        """Unpacks all datasets.
        You can call dict(this_instance) then to get a real dict.
        """
        load(self, lazy=False)

    def close(self):
        """Closes the h5file if provided at initialization."""
        if self._h5file and hasattr(self._h5file, 'close'):
            self._h5file.close()      

    def __del__(self):
        self.close()

    def _ipython_key_completions_(self):
        """Returns a tuple of keys. 
        Special Method for ipython to get key completion
        """
        return tuple(self.keys())
            

def load(hdf, lazy=True, unpacker=unpack_dataset, mode='r', *args, **kwargs):
    """Returns a dictionary containing the
    groups as keys and the datasets as values
    from given hdf file.

    Parameters
    ----------
    hdf : string (path to file) or `h5py.File()` or `h5py.Group()`
    lazy : bool
        If True, the datasets are lazy loaded at the moment an item is requested.
    upacker : callable
        Unpack function gets `value` of type h5py.Dataset.
        Must return the data you would like to have it in the returned dict.
    mode : str
        File read mode. Default: 'r'.

    Returns
    -------
    d : dict
        The dictionary containing all groupnames as keys and
        datasets as values, with group/file attributes under 'attrs'.
    """

    def _decode_attr(v):
        # Decode bytes to str if needed
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8")
            except Exception:
                return v
        return v

    def _recurse(hdfobject, datadict):
        # Extract attributes
        attrs = {}
        for k, v in hdfobject.attrs.items():
            attrs[k] = _decode_attr(v)
        if attrs:
            datadict['attrs'] = attrs

        for key, value in hdfobject.items():
            if isinstance(value, h5py.Group):
                if lazy:
                    datadict[key] = LazyHdfDict()
                else:
                    datadict[key] = {}
                datadict[key] = _recurse(value, datadict[key])
            elif isinstance(value, h5py.Dataset):
                if not lazy:
                    value = unpacker(value)
                datadict[key] = value

        return datadict

    with hdf_file(hdf, lazy=lazy, mode=mode, *args, **kwargs) as hdf:
        if lazy:
            data = LazyHdfDict(_h5file=hdf)
        else:
            data = {}
        return _recurse(hdf, data)


def pack_dataset(hdfobject, key, value):
    """Packs a given key value pair into a dataset in the given hdfobject."""

    isdt = None
    if isinstance(value, datetime):
        value = value.timestamp()
        isdt = True

    if hasattr(value, '__iter__'):
        if all(isinstance(i, datetime) for i in value):
            value = [item.timestamp() for item in value]
            isdt = True

    try:
        ds = hdfobject.create_dataset(name=key, data=value)
        
        if isdt:
            attr_data = "datetime"
        elif isinstance(value, list):
            attr_data = "list"
        elif isinstance(value, tuple):
            attr_data = "tuple"
        elif isinstance(value, str):
            attr_data = "str"
            value = value.encode("utf-8")  # encode string to bytes
        else:   
            attr_data = None

        if attr_data:
            ds.attrs.create(name=TYPE, data=str_(attr_data))

    except (TypeError, ValueError):
        # Obviously the data was not serializable. To give it
        # a last try; serialize it to yaml
        # and save it to the hdf file:
        ds = hdfobject.create_dataset(
            name=key,
            data=str_(yaml.safe_dump(value)))

        ds.attrs.create(name=TYPE, data=str_("yaml"))
        # if this fails again, restructure your data!   


def dump(data, hdf, packer=pack_dataset, mode='w', *args, **kwargs):
    """Adds keys of given dict as groups and values as datasets
    to the given hdf-file (by string or object) or group object.

    Parameters
    ----------
    data : dict
        The dictionary containing only string keys and
        data values or dicts again.
    hdf : string (path to file) or `h5py.File()` or `h5py.Group()`
    packer : callable
        Callable gets `hdfobject, key, value` as input.
        `hdfobject` is considered to be either a h5py.File or a h5py.Group.
        `key` is the name of the dataset.
        `value` is the dataset to be packed and accepted by h5py.
    mode : str
        File write mode. Default: 'w'

    Returns
    -------
    hdf : obj
        `h5py.Group()` or `h5py.File()` instance

    """

    def _recurse(datadict, hdfobject):
        # Handle attributes if present
        attrs = datadict.pop('attrs', None) if isinstance(datadict, dict) else None
        if attrs is not None:
            for k, v in attrs.items():
                if isinstance(v, str):
                    hdfobject.attrs[k] = v.encode("utf-8")
                elif isinstance(v, (int, float)):
                    hdfobject.attrs[k] = v
                else:
                    raise TypeError(f"Attribute '{k}' must be str, int, or float, not {type(v)}")
        # Handle groups and datasets
        for key, value in datadict.items():
            if isinstance(value, (dict, LazyHdfDict)):
                hdfgroup = hdfobject.create_group(key)
                _recurse(value, hdfgroup)
            else:
                packer(hdfobject, key, value)

    with hdf_file(hdf, mode=mode, *args, **kwargs) as hdf:
        _recurse(data, hdf)
    
#%%
if __name__ == "__main__":
    # Example usage
    #%%
    import numpy as np
    data = {
        'attrs': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "This is an example dataset.",
        },
        'group1': {
            'dataset1': np.array([1, 2, 3]),
            # 'dataset2': 'example string',
            # 'dataset3': datetime.now(),
        },
        'group2': {
            'nested_group': {
                'nested_dataset': np.array([1, 2, 3]),
            }
        }
    }

    dump(
        data, 
        'example.h5', 
    )
    loaded_data = load('example.h5')
