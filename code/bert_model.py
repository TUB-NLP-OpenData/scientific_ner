import json
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score
import re
from collections import namedtuple
from typing import List, Tuple, NamedTuple
import pandas as pd
import json
import numpy as np
from flair.data import iob2, iob_iobes
from sklearn import metrics
import transformers
from transformers import BertForTokenClassification, AdamW
transformers.__version__
from seqeval.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup


df = pd.read_json (r'file_test.json1', lines=True)


BIO = {"B", "I", "O"}
BIOES = {"B", "I", "O", "E", "S"}
Sequences = List[List[str]]


def calc_seqtag_f1_scores(
    predictions: Sequences, targets: Sequences,
):
    assert set([t[0] for s in targets for t in s]).issubset(BIO)
    assert set([t[0] for s in predictions for t in s]).issubset(BIO)
    assert all([len(t) == len(p) for t, p in zip(targets, predictions)])
    _, _, f1_train = spanlevel_pr_re_f1(predictions, targets)
    # tokenlevel_scores = calc_seqtag_tokenlevel_scores(targets, predictions)
    return {
        # "token-level": tokenlevel_scores,
        "f1-micro-spanlevel": f1_train,
        "seqeval-f1": f1_score(targets, predictions),
    }


def mark_text(text, char_spans):
    sorted_spans = sorted(char_spans, key=lambda sp: -sp[0])
    for span in sorted_spans:
        assert span[1] > span[0]
        text = text[: span[1]] + "</" + span[2] + ">" + text[span[1] :]
        text = text[: span[0]] + "<" + span[2] + ">" + text[span[0] :]
    return text


def correct_biotags(tag_seq):
    correction_counter = 0
    corr_tag_seq = tag_seq
    for i in range(len(tag_seq)):
        if i > 0 and tag_seq[i - 1] is not "O":
            previous_label = tag_seq[i - 1][2:]
        else:
            previous_label = "O"
        current_label = tag_seq[i][2:]
        if tag_seq[i].startswith("I-") and not current_label is not previous_label:
            correction_counter += 1
            corr_tag_seq[i] = "B-" + current_label
    return corr_tag_seq


def iob2iobes(tags: List[str]):
    Label = namedtuple("Label", "value")  # just to please flair
    tags = [Label(tag) for tag in tags]
    iob2(tags)
    tags = iob_iobes(tags)
    return tags


def bilou2bio(tag_seq):
    """
    BILOU to BIO
    or
    BIOES to BIO
    E == L
    S == U
    """
    bio_tags = tag_seq
    for i in range(len(tag_seq)):
        if tag_seq[i].startswith("U-") or tag_seq[i].startswith("S-"):
            bio_tags[i] = "B-" + tag_seq[i][2:]
        elif tag_seq[i].startswith("L-") or tag_seq[i].startswith("E-"):
            bio_tags[i] = "I-" + tag_seq[i][2:]
    assert set([t[0] for t in bio_tags]).issubset(BIO), set([t[0] for t in bio_tags])
    return bio_tags


def spanlevel_pr_re_f1(label_pred, label_correct):
    """
    see: https://github.com/UKPLab/deeplearning4nlp-tutorial/blob/master/2015-10_Lecture/Lecture3/code/BIOF1Validation.py
    """
    pred_counts = [
        compute_TP_P(pred, gold) for pred, gold in zip(label_pred, label_correct)
    ]
    gold_counts = [
        compute_TP_P(gold, pred) for pred, gold in zip(label_pred, label_correct)
    ]
    prec = np.sum([x[0] for x in pred_counts]) / np.sum([x[1] for x in pred_counts])
    rec = np.sum([x[0] for x in gold_counts]) / np.sum([x[1] for x in gold_counts])
    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def calc_seqtag_tokenlevel_scores(gold_seqs: Sequences, pred_seqs: Sequences):
    gold_flattened = [l for seq in gold_seqs for l in seq]
    pred_flattened = [l for seq in pred_seqs for l in seq]
    assert len(gold_flattened) == len(pred_flattened) and len(gold_flattened) > 0
    label_set = list(set(gold_flattened + pred_flattened))
    scores = {
        "f1-micro": metrics.f1_score(gold_flattened, pred_flattened, average="micro"),
        "f1-macro": metrics.f1_score(gold_flattened, pred_flattened, average="macro"),
        "cohens-kappa": metrics.cohen_kappa_score(gold_flattened, pred_flattened),
        "clf-report": metrics.classification_report(
            gold_flattened,
            pred_flattened,
            target_names=label_set,
            digits=3,
            output_dict=True,
        ),
    }
    return scores


def compute_TP_P(guessed, correct):
    """
    see: https://github.com/UKPLab/deeplearning4nlp-tutorial/blob/master/2015-10_Lecture/Lecture3/code/BIOF1Validation.py
    """
    assert len(guessed) == len(correct)
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == "B":  # A new chunk starts
            count += 1

            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True

                while (
                    idx < len(guessed) and guessed[idx][0] == "I"
                ):  # Scan until it no longer starts with I
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == "I":  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    return correctCount, count


def char_precise_spans_to_token_spans(
    char_spans: List[Tuple[int, int, str]], token_spans: List[Tuple[int, int]]
):
    spans = []
    for char_start, char_end, label in char_spans:
        closest_token_start = int(
            np.argmin(
                [np.abs(token_start - char_start) for token_start, _ in token_spans]
            )
        )
        closest_token_end = int(
            np.argmin([np.abs(token_end - char_end) for _, token_end in token_spans])
        )
        spans.append((closest_token_start, closest_token_end, label))
    return spans


def char_precise_spans_to_BIO_tagseq(
    char_precise_spans: List[Tuple[int, int, str]], start_ends: List[Tuple[int, int]]
) -> List[str]:
    tags = ["O" for _ in range(len(start_ends))]

    def find_closest(seq: List[int], i: int):
        return int(np.argmin([np.abs(k - i) for k in seq]))

    for sstart, send, slabel in char_precise_spans:
        closest_token_start = find_closest([s for s, e in start_ends], sstart)
        closest_token_end = find_closest([e for s, e in start_ends], send)
        if closest_token_end - closest_token_start == 0:
            tags[closest_token_start] = "B-" + slabel
        else:
            tags[closest_token_start] = "B-" + slabel
            tags[closest_token_end] = "I-" + slabel
            for id in range(closest_token_start + 1, closest_token_end):
                tags[id] = "I-" + slabel
    return tags



def regex_tokenizer(
    text, pattern=r"(?u)\b\w\w+\b"
) -> List[Tuple[int, int, str]]:  # pattern stolen from scikit-learn
    return [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, text)]


def minimal_test_spans_to_bio_tagseq(text, spans):


    return_text = []
    return_labels = []
    #text = "xxx xxy yy oyo"
    #spans = [(0, 5, "X"), (6, 9, "Y"), (12, 12, "Y")]
    tokens = regex_tokenizer(text)
    tags = char_precise_spans_to_BIO_tagseq(
        spans, start_ends=[(s, e) for s, e, t in tokens]
    )
    #print("original labeled spans")
    #for s, e, l in spans:
    #    print("%s\t%s" % (text[s : (e + 1)], l))

    #print("more or less messed up labeles due to tokenizing")
    for (_, _, tok), tag in zip(tokens, tags):
        #print("%s\t%s" % (tok, tag))
        #return_text.append(tok)
        return_labels.append(tag)

    return return_labels#,return_text





print ()
## Get pairs of sentences and their BIO tags

sentences = []
span_labels = []

#for i in range(df.shape[0]):

#    actual_text = df.iloc[i,4]
#    actual_labels = df.iloc[i,2]

for index, row in df.iterrows():
    #print (row.keys())
    actual_text = row['text']
    actual_labels = row['labels']
    #print (actual_text)
    #print (actual_labels)

    pair_of_text_label = [actual_text,minimal_test_spans_to_bio_tagseq(actual_text,actual_labels)]

    sentences.append(pair_of_text_label[0])
    span_labels.append(pair_of_text_label[1])

data = {'sentences': sentences, 'labels':span_labels}
df1 = pd.DataFrame(data)

#print (len(df1))

# Convert to tokenization system supported by BERT

labels = span_labels

MAX_LEN = 64

device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]



# Map uniques tags to indices to define output

flat_labels = [item for sublist in labels for item in sublist]

tag_values = list(set(flat_labels))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}



# Tokenize according to format accepted by BERT

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")





tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")



# BERT specific initialization

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


# Split train and test

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, test_size=0.3)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=0.3)


# Convert to pytorch tensors

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)



# Initialze dataloader

#Batch size
bs = 32

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)






model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)


FULL_FINETUNING = False
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

# Use Adam optimizer
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)




epochs = 10
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)




## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()
