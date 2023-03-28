# Importing relevant modules
import spacy
import PyPDF2
import sys
import os
import pandas as pd
import numpy as np
import torch
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging

# Setting logging level
logging.set_verbosity_error()

# Defining PyTorch Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Defining Coreference Resolution modular function
def resolve_references(doc):
    token_mention_mapper = {}
    output_string = ''
    clusters = [
        val for key, val in doc.spans.items() if key.startswith('coref_cluster')
    ]

    # Iterating through each cluster
    for cluster in clusters:
        first_mention = cluster[0]

        # Marking spans for references
        for mention_span in list(cluster)[1:]:
            if len(mention_span) == 0:
                continue
            else:
                token_mention_mapper[mention_span[0].idx] = (
                    first_mention.text + mention_span[0].whitespace_
                )
                for token in mention_span[1:]:
                    token_mention_mapper[token.idx] = ''

    # Recreating sentences after resolution
    for token in doc:
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        else:
            output_string += token.text + token.whitespace_

    return output_string

# Defining central pipeline for extracting goals
def extract_goals(input_text):
    global s
    
    # First stage - Coreference Resolution
    coref_model = spacy.load('en_coreference_web_trf')
    coref_model.max_length = 2000000

    # Performing inference on Coreference Resolution transformer
    doc = coref_model(input_text)
    resolved_string = resolve_references(doc)

    # Manual garbage collection
    del coref_model, doc

    # Loading core Spacy transformer model
    spacy_core = spacy.load('en_core_web_trf')
    doc = spacy_core(resolved_string)
    
    # Second and third stage - NER and POS tagging
    action_statements = list()
    flag = False
    list_of_sent = list(doc.sents)
    
    # Iterating through each sentence, to check if its an action statement
    for sent in list_of_sent:
        flag = False
        sent = list(sent)
        
        # Checking if an orgnaization is followed by an action word
        for word in sent:
            if word.ent_type_ == 'ORG':
                location = sent.index(word)
                if (location + 1 != len(sent)) and sent[location + 1].tag_ in (
                    'VB',
                    'VBG',
                    'VBP',
                    'VBZ',
                    'MD',
                    'VV',
                    'VP',
                    'VERB',
                    'VAFIN',
                    'VMFIN',
                    'VVFIN',
                    'VE'
                ):
                    flag = True
                    break

        # If action statement is found its collected
        if flag:
            str_sent = list(map(lambda x: str(x), sent))
            action_statements.append(' '.join(str_sent).replace(' - ', ' '))

    # Additional info to be added in the report
    s += f'Total no. of action statements: {len(action_statements)}'+'\n'
    s += f'Total no. of sentences: {len(list_of_sent)}'+'\n\n'
    
    # Manual garbage collection
    del spacy_core, doc
    
    # Loading compatible tokenizer
    base_model = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Tokenizing action_statments
    action_statements_tokenized = tokenizer(
        action_statements, padding=True, truncation=True, max_length=512
    )

    # Converting tokenized sentences to PyTorch Dataset
    test_dataset = Dataset(action_statements_tokenized)

    # Loading fine-tuned BERT model
    model_path = 'models/mar11/checkpoint-180'
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    test_trainer = Trainer(model)

    # Performing inference on fine-tuned BERT model
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    y_pred = np.argmax(raw_pred, axis=1)

    # Converting list to pandas Series
    y_pred = pd.Series(y_pred)

    # Getting index of action statments which we predicted as goals
    goal_indices = list(y_pred[y_pred==1].index)
    
    # Preparing list of goals to be returned
    goals = list()
    for i in goal_indices:
        goals.append(action_statements[i])
    return goals

# Starting point of the script when executed as a program
if __name__ == "__main__":
    # Getting the mode from CLI arguments
    mode = sys.argv[1]

    # Interactive mode
    if mode == 'i':
        # Joining the CLI argument
        input_text = ' '.join(sys.argv[2:])

        # Running the complete goal extraction pipeline
        s = ''''''
        goals = extract_goals(input_text)
        
        # Constructing output file
        n = 0
        for i in goals:
            n += 1
            s += f'Goal {n}: ' + '\n'
            s += i + '\n'
        s = f'Total no. of goals extracted: {n}' + '\n' + s

        # Write to output file
        with open('output/temp.txt', 'w') as f:
            f.write(s)

    # PDF file mode 
    elif mode == 'f':
        # Reading the pdf file from CLI argument
        file_path = sys.argv[2]
        reader = PyPDF2.PdfReader(file_path)
        input_text = ''
        for i in reader.pages:
            input_text += i.extract_text()
        input_text = input_text.replace('\n', ' ')
        
        # Running the complete goal extraction pipeline
        s = ''''''
        goals = extract_goals(input_text)
        
        # Constructing output file
        n = 0
        for i in goals:
            n += 1
            s += f'Goal {n}: ' + '\n'
            s += i + '\n'
        s = f'Total no. of goals extracted: {n}' + '\n\n' + s

        # Write to output file
        with open('output/temp.txt', 'w') as f:
            f.write(s)
    
    elif mode == 'd':
        # Scanning files in the given directory
        dir_path = sys.argv[2]
        list_of_all_file = os.listdir(dir_path)
        
        # Ignoring files that are not pdf
        for i in list_of_all_file:
            if not i.endswith('.pdf'):
                list_of_all_file.remove(i)
        
        print(f'Reading files: {list_of_all_file}')

        # Reading from each pdf in the directory
        for i in list_of_all_file:
            file_path = dir_path+'/'+i
            reader = PyPDF2.PdfReader(file_path)
            input_text = ''
            for j in reader.pages:
                input_text += j.extract_text()
            input_text = input_text.replace('\n', ' ')
            
            # Running the complete goal extraction pipeline
            s = ''''''
            goals = extract_goals(input_text)
            
            # Constructing output file
            n = 0
            for j in goals:
                n += 1
                s += f'Goal {n}: ' + '\n'
                s += j + '\n'
            s = f'Total no. of goals extracted: {n}' + '\n\n' + s

            # Write to output file
            with open('output/'+i[:-4]+'_result.txt', 'w') as f:
                f.write(s)

    # In case of invalid input
    else:
        print(f'invalid input method')
