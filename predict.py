import torch
from vars import full_emotion_labels, MIN_WORD_FREQUENCY
from dataset import TWEET_SET
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import argparse
from vars import csvdir

emotion_labels = full_emotion_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('spacy',language='en_core_web_sm')
train_iter = TWEET_SET(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"], min_freq=MIN_WORD_FREQUENCY)
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

train_iter = TWEET_SET(split='train')
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64

parser = argparse.ArgumentParser(description='Prediction.')
parser.add_argument('text', action='store', type=str, help='The text to parse.')
parser.add_argument('id', action='store', type=str, help='The tweet id.')
args = parser.parse_args()

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1, output[0]

model = torch.load(csvdir + '/model.pt')
prediction = predict(args.text, text_pipeline)
tensor_output = {label[1]:float(prediction[1][int(label[0])-1]) for label in emotion_labels.items()}


import matplotlib.pyplot as plt
from matplotlib import style

step1 = 0.03571428571428571
step2 = 0.029411764705882353

step = step2

c = [
        (1-step*i,
         1-step*i,
         1-step*i) for i in range(1, 35)
    ]

data = {key: value for key, value in sorted(tensor_output.items(), key=lambda item: item[1])}

print(data)

emotions = list(data.keys())
coefficient = list(data.values())
plt.style.use('seaborn-whitegrid')
plt.bar(range(len(data)), coefficient, tick_label=emotions, color=c)
plt.xticks(rotation='vertical')
plt.title('Emotion Label')
plt.savefig('{}.svg'.format(args.id),
            dpi=500,
            transparent=True,
            bbox_inches="tight")
