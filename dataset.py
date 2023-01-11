from torchtext.data.datasets_utils import _wrap_split_argument
import os
import logging
import hashlib
import io
import requests
import re
import sys
import csv
import torch
import inspect
import importlib.util
from vars import csvdir
from tqdm import tqdm


os.chdir(csvdir)


def is_module_available(*modules: str) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in modules)

def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get('Content-length', 0))
    with tqdm(total=total_size, unit='B', unit_scale=1) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                t.update(len(chunk))
                yield chunk


def _get_response_from_google_drive(url):
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
    if confirm_token is None:
        if "Quota exceeded" in str(response.content):
            raise RuntimeError(
                "Google drive link {} is currently unavailable, because the quota was exceeded.".format(
                    url
                ))
        else:
            raise RuntimeError("Internal error: confirm_token was not found in Google drive link.")

    url = url + "&confirm=" + confirm_token
    response = session.get(url, stream=True)

    if 'content-disposition' not in response.headers:
        raise RuntimeError("Internal error: headers don't contain content-disposition.")

    filename = re.findall("filename=\"(.+)\"", response.headers['content-disposition'])
    if filename is None:
        raise RuntimeError("Filename could not be autodetected")
    filename = filename[0]

    return response, filename


class DownloadManager:
    def get_local_path(self, url, destination):
        if 'drive.google.com' not in url:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        else:
            response, filename = _get_response_from_google_drive(url)

        with open(destination, 'wb') as f:
            for chunk in _stream_response(response):
                f.write(chunk)


_DATASET_DOWNLOAD_MANAGER = DownloadManager()

def _dataset_docstring_header(fn, num_lines=None, num_classes=None):
    """
    Returns docstring for a dataset based on function arguments.

    Assumes function signature of form (root='.data', split=<some tuple of strings>, **kwargs)
    """
    argspec = inspect.getfullargspec(fn)
    if not (argspec.args[0] == "root" and
            argspec.args[1] == "split"):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))
    default_split = argspec.defaults[1]

    if not (isinstance(default_split, tuple) or isinstance(default_split, str)):
        raise ValueError("default_split type expected to be of string or tuple but got {}".format(type(default_split)))

    header_s = fn.__name__ + " dataset\n"

    if isinstance(default_split, tuple):
        header_s += "\nSeparately returns the {} split".format("/".join(default_split))

    if isinstance(default_split, str):
        header_s += "\nOnly returns the {} split".format(default_split)

    if num_lines is not None:
        header_s += "\n\nNumber of lines per split:"
        for k, v in num_lines.items():
            header_s += "\n    {}: {}\n".format(k, v)

    if num_classes is not None:
        header_s += "\n\nNumber of classes"
        header_s += "\n    {}\n".format(num_classes)

    args_s = "\nArgs:"
    args_s += "\n    root: Directory where the datasets are saved."
    args_s += "\n        Default: .data"

    if isinstance(default_split, tuple):
        args_s += "\n    split: split or splits to be returned. Can be a string or tuple of strings."
        args_s += "\n        Default: {}""".format(str(default_split))

    if isinstance(default_split, str):
        args_s += "\n     split: Only {default_split} is available."
        args_s += "\n         Default: {default_split}.format(default_split=default_split)"

    return "\n".join([header_s, args_s]) + "\n"

def _add_docstring_header(docstring=None, num_lines=None, num_classes=None):
    def docstring_decorator(fn):
        old_doc = fn.__doc__
        fn.__doc__ = _dataset_docstring_header(fn, num_lines, num_classes)
        if docstring is not None:
            fn.__doc__ += docstring
        if old_doc is not None:
            fn.__doc__ += old_doc
        return fn
    return docstring_decorator

class _RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, description, full_num_lines, iterator):
        """Initiate the dataset abstraction.
        """
        super(_RawTextIterableDataset, self).__init__()
        self.description = description
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return self.description


def reporthook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def validate_file(file_obj, hash_value, hash_type="sha256"):

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    elif hash_type == "md5":
        hash_func = hashlib.md5()
    else:
        raise ValueError

    while True:
        chunk = file_obj.read(1024 ** 2)
        if not chunk:
            break
        hash_func.update(chunk)
    return hash_func.hexdigest() == hash_value


def _check_hash(path, hash_value, hash_type):
    logging.info('Validating hash {} matches hash of {}'.format(hash_value, path))
    with open(path, "rb") as file_obj:
        if not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(os.path.abspath(path)))


def download_from_url(url, path=None, root='.data', overwrite=False, hash_value=None,
                      hash_type="sha256"):

    if path is None:
        _, filename = os.path.split(url)
        root = os.path.abspath(root)
        path = os.path.join(root, filename)
    else:
        path = os.path.abspath(path)
        root, filename = os.path.split(os.path.abspath(path))

    if os.path.exists(path):
        logging.info('File %s already exists.' % path)
        if not overwrite:
            if hash_value:
                _check_hash(path, hash_value, hash_type)
            return path

    if not os.path.exists(root):
        try:
            os.makedirs(root)
        except OSError:
            raise OSError("Can't create the download directory {}.".format(root))

    _DATASET_DOWNLOAD_MANAGER.get_local_path(url, destination=path)

    logging.info('File {} downloaded.'.format(path))

    if hash_value:
        _check_hash(path, hash_value, hash_type)

    return path


def unicode_csv_reader(unicode_csv_data, **kwargs):
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line



URL = {
    'train': "https://www.j0lms.com/static/datasets/train.csv",
    'test': "https://www.j0lms.com/static/datasets/test.csv",
}

NUM_LINES = {
    'train': 120000,
    'test': 7600,
}


@_add_docstring_header(num_lines=NUM_LINES, num_classes=4)
@_wrap_split_argument(('train', 'test'))
def TWEET_SET(root, split):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            next(reader)
            for row in reader:
                yield int(row[0]), ' '.join(row[1:])

    path = download_from_url(URL[split], root=root,
                             overwrite=True,
                             path=os.path.join(root, split + ".csv")
                             )
    return _RawTextIterableDataset("TWEET_SET", NUM_LINES[split],
                                   _create_data_from_csv(path))
