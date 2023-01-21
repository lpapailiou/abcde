# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:46:54 2023

@author: Lena Papailiou
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:54:33 2023

@author: Lena Papailiou

!pip install contractions
!git clone https://github.com/huggingface/neuralcoref.git
!pip install -U spacy~=2.3.5
!python -m spacy download en
!python -m spacy download en_core_web_sm

%cd neuralcoref

!pip install -r requirements.txt
!pip install -e .

"""
import os
import time
import json
import neuralcoref
import spacy
import nltk
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions
from transformers import pipeline

# ============== contraction ============== #

def expand(text):
    expanded_words = [] 
    for word in text.split():
        expanded_words.append(contractions.fix(word))  
    return' '.join(expanded_words)

# ============== lemmatization ============== #
  
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
  tokenized = word_tokenize(text)
  lemmatized = [lemmatizer.lemmatize(item) for item in tokenized]
  return ' '.join(lemmatized)


# ============== resolve coreferences ============== #

nlpraw = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def resolve_coref(data_str):
    resolved = nlp(data_str)._.coref_resolved
    resolved = resolved.split('\n')
    if '' in resolved: resolved.remove('')
    if '.' in resolved: resolved.remove('.')
    if ' ' in resolved: resolved.remove(' ')
    if '. ' in resolved: resolved.remove('. ')
    return resolved
    
# ============== sentiment parsing ============== #

sentiment_pipeline = pipeline("sentiment-analysis",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)

def get_clause(text):
    # https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python
    doc = nlpraw(text)    
    seen = set() # keep track of covered words
    
    chunks = []
    for sent in doc.sents:
        heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']
    
        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = (' '.join([ww.text for ww in words]))
            chunks.append( (head.i, chunk) )
    
        unseen = [ww for ww in sent if ww not in seen]
        chunk = ' '.join([ww.text for ww in unseen])
        chunks.append( (sent.root.i, chunk) )
    
    chunks = sorted(chunks, key=lambda x: x[0])
    out = []
    for i in range(0, len(chunks)):
        out.append(chunks[i][1])
    return out

def merge_dict(dic, src):
    items = []
    for j in range(0,len(dic)):
        item = max(dic[j], key=lambda x:x['score'])
        if isinstance(src[j], str):
            item["text"] = src[j]
        else:
            item["text"] = src[j][1]
        items.append(item)
    return items


sentiment_segment_total = 0
sentiment_segment_removed = 0

def remove_sentiment(text):
    global sentiment_segment_total
    global sentiment_segment_removed
    src = get_clause(text)
    segments = len(src)
    sentiment_segment_total += segments
    if segments < 2:
        return text
    sentiments = sentiment_pipeline(src)
    merged = merge_dict(sentiments, src)
    out = []
    out.append(min(merged, key=lambda x:x['score'])["text"])
    sentiment_segment_removed = segments - len(out)
    return ' '.join(out)

# ============== pipeline ============== #

def execute_pipeline(args, datasets):
    print("---start of preprocessing---")
    start_time = time.time()
    dialog_count = 0
    error_count = 0
    line_count = 0
    altered_line_count = 0
    for dataset in datasets:
        if type(dataset) is not dict:
            dataset = datasets[dataset]
        for i in range(0, len(dataset)):
            # collect conversation
            delexed_data = dataset[i]["delexed"]
            
            # expand
            if args.expand == True:
                for j in range(0,len(delexed_data)):
                    delexed_data[j]['text'] = expand(delexed_data[j]['text'])
                    
            # coref
            if args.coref == True:
                delexed = ''
                for j in range(0,len(delexed_data)):
                    delexed += delexed_data[j]['text']
                    if j < len(delexed_data)-1:
                        delexed += "\n"
                delexed_res = resolve_coref(delexed)
                dialog_count += 1
                line_count += len(delexed_data)
                if (len(delexed_data) == len(delexed_res)):
                    for j in range(0,len(delexed_data)):
                        if (delexed_data[j]['text'] != delexed_res[j]):
                            altered_line_count += 1
                            delexed_data[j]['text'] = delexed_res[j]  
                else:
                    error_count += 1
						
            for j in range(0,len(delexed_data)):
                # sentiment parsing
                if args.sentiment == True:
                    delexed_data[j]['text'] = remove_sentiment(delexed_data[j]['text'])  
                # lemmatize 
                if args.lemmatize == True:
                    delexed_data[j]['text'] = lemmatize_text(delexed_data[j]['text'])  		  
			  
    print(str(error_count) + ' of ' + str(dialog_count) + ' dialogs not processed in coref parsing')
    print(str(altered_line_count) + ' of ' + str(line_count) + ' lines altered in coref parsing')
    print(str(sentiment_segment_removed) +' of ' + str(sentiment_segment_total) + ' segments removed in sentiment parsing')
    print("---preprocessing terminated, took %s seconds ---" % (time.time() - start_time))    
    return datasets

def preprocess_dialog_data(args, datasets):
    if args.expand == True or args.coref == True or args.sentiment == True or args.lemmatize == True:
        checkpoint_folder = f'{args.prefix}_{args.filename}_{args.model_type}_{args.suffix}_data'
        ckpt_dir = os.path.join(args.output_dir, args.task, checkpoint_folder)
        if os.path.exists(ckpt_dir):
            print(ckpt_dir + ' exists')
        else:
            os.mkdir(ckpt_dir)
        dir = os.listdir(ckpt_dir)
        filepath = os.path.join(ckpt_dir, 'dump.json')
        if (len(dir) != 0):
            print('load preprocessed data from directory: ' + ckpt_dir)
            datasets = json.load(open(filepath, 'r'))
        else:
            print('perform preprocessing: ' + ' expand ' + str(args.expand)+ ' | coref ' + str(args.coref)+ ' | sentiment ' + str(args.sentiment)+ ' | lemmatize ' + str(args.lemmatize))
            datasets = execute_pipeline(args, datasets)
            print('save preprocessed dump to: ' + filepath)
            json_object = json.dumps(datasets, indent=4)
            with open(filepath, "w") as outfile:
                outfile.write(json_object)
    return datasets
    
    
    
