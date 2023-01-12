from fastapi import FastAPI, Form

import spacy
from spacy.tokens import Doc
import re


app = FastAPI()
coref_model = spacy.load('en_coreference_web_trf')

def resolve_references(doc):
    token_mention_mapper = {}
    output_string = ''
    clusters = [val for key, val in doc.spans.items() if key.startswith('coref_cluster')]

    for cluster in clusters:
        first_mention = cluster[0]
        for mention_span in list(cluster)[1:]:
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
            for token in mention_span[1:]:
                token_mention_mapper[token.idx] = ''

    for token in doc:
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        else:
            output_string += token.text + token.whitespace_

    return output_string

def show_ents(doc):
    global ner_op_string
    
    if doc.ents:
        
        for ent in doc.ents:
            # print(ent.text+' - ' +str(ent.start_char) +' - '+ str(ent.end_char) +' - '+ent.label_+ ' - '+str(spacy.explain(ent.label_)))
            
            if ent.label_ == 'ORG':
                word_length = ent.end_char - ent.start_char
                ner_op_string = ner_op_string[:ent.start_char] + 'O'*(word_length) + ner_op_string[ent.end_char:]
    
    else:
        print('No named entities found.')
    
    return ner_op_string

@app.get("/")
async def get_goals(input_text):
    doc = coref_model(input_text)
    return {"goals": resolve_references(doc)}
