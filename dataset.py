from torchtext.utils import download_from_url, unicode_csv_reader
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _wrap_split_argument
from torchtext.data.datasets_utils import _add_docstring_header
import os
import io
from vars import csvdir

os.chdir(csvdir)

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
                             path=os.path.join(root, split + ".csv")
                             )
    return _RawTextIterableDataset("TWEET_SET", NUM_LINES[split],
                                   _create_data_from_csv(path))
