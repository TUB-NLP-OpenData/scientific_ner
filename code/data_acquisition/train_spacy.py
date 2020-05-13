import pickle
import spacy
import random
import json
from math import floor

def get_training_and_testing_sets(file_list, split = 0.2):

    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

# TRAIN_DATA = pickle.load(open("../Annotations_doccano/final_corrected/final_data_corrected.json","rb"))
# print(len(TRAIN_DATA))
#


from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(ner_model, examples):
    scorer = Scorer()
    for sents, ents in examples:
        try:
            doc_gold = ner_model.make_doc(sents)
            entity = ([(ent[0], ent[1], ent[2]) for ent in ents['entities']])
            #print (entity)
            #print (ents['entities'])
            gold = GoldParse(doc_gold, entities=entity)
            pred_value = ner_model(sents)
            scorer.score(pred_value, gold)
        except:
            pass
    print (scorer.scores)
    return scorer.scores



def get_json(file_name):
    print (file_name)
    data = []
    for file in file_name:
        with open(file) as f:
            for line in f:
                #print (line)
                row=json.loads(line)
                data.append((row["text"], {"entities": row["labels"]}) )
    return data
#TRAIN_DATA=get_json(["../Annotations_doccano/final_corrected/final_data_corrected.json"])
#TRAIN_DATA=get_json(["../original_data/original_data.json"])
TRAIN_DATA=get_json(["../Annotations_doccano/final_corrected/final_data_corrected.json","../original_data/original_data.json"])
random.shuffle(TRAIN_DATA)
#TRAIN_DATA=TRAIN_DATA[:300]
#print (TRAIN_DATA)
#new_list = TRAIN_DATA[:]

def train_spacy(data, iterations):
    TRAIN_DATA, test = get_training_and_testing_sets(data,0.8)
    print (len(TRAIN_DATA))
    print (len(test))
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):

            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            n_entities=0

            for text, annotations in TRAIN_DATA:
                n_entities+=len(annotations['entities'])
                #print (annotations['entities'])
                #print (n_entities)
                try:
                    nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
                except:
                    pass
            print(losses)
            evaluate(nlp, test)

    return nlp,counter


prdnlp,counter = train_spacy(TRAIN_DATA, 50)

print(counter)
# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

# Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
