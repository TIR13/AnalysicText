# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:52:40 2021

@author: Адм
"""

import os
import random
import spacy
from spacy.util import minibatch, compounding
#from spacy.pipeline.textcat import single_label_cnn_config

#print(spacy.__version__)
def load_training_data(split=0.8, limit=0):
    
    # Загрузка данных из файлов
    reviews = []    
    filename = "train.csv"
    with open(filename, encoding = 'utf-8') as f:
        #cnt=0
        for line in f:
            #cnt+=1
            #if cnt > 10: break
            try:
                line=line.strip()
                text, pos=line.split("\t")
                #print(text, pos)
                if pos == "neautral": continue
                spacy_label={
                    "cats": {
                        "pos": "positive" == pos,
                        "neg": "negative" == pos        
                    }
                }
                reviews.append((text,spacy_label))
            except ValueError:
                #print(f"Error load data: {ValueError}")
                pass
                #break
            #print(line)
    # перемешали записи из данных
    print(len(reviews))
    random.shuffle(reviews)
    if limit and limit <= len(reviews):
        reviews=reviews[:limit]
    split = int(len(reviews) * split)          #
    return reviews[:split], reviews[split:] 

def evaluate_model(tokenizer, textcat, test_data: list):
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, чтобы в знаменателе
    # не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        score_pos = review.cats['pos'] 
        if true_label['pos']:
            if score_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if score_pos >= 0.5:
                FP += 1
            else:
                TN += 1    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

def train_model(training_data: list, test_data: list, iterations: int = 20):  
    # Строим конвейер
    nlp = spacy.load('ru2')
    
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe("textcat", config={"architecture": "simple_cnn"})
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    textcat.add_label("pos")
    textcat.add_label("neg")

    # Обучаем только textcat
    training_excluded_pipes = [ pipe for pipe in nlp.pipe_names if pipe != "textcat" ]
    with nlp.disable_pipes(training_excluded_pipes):          #
        optimizer = nlp.begin_training()                      #
        # Итерация обучения                           
        print("Начинаем обучение")                            #
        batch_sizes = compounding( 4.0, 32.0, 1.001)    
        for i in range(iterations):                           
            loss = {}                                         
            random.shuffle(training_data)                     
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:                             
                text, labels = zip(*batch)                    
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
                with textcat.model.use_params(optimizer.averages):
                    evaluation_results = evaluate_model(   
                        tokenizer=nlp.tokenizer,           
                        textcat=textcat,                   
                        test_data=test_data
                        )                  
                    
                    print(f"{loss['textcat']:9.6f}\t\
                        {evaluation_results['precision']:.3f}\t\
                        {evaluation_results['recall']:.3f}\t\
                        {evaluation_results['f-score']:.3f}")
    # Сохраняем модель                                 #
    with nlp.use_params(optimizer.averages):           #
        nlp.to_disk("model_artifacts") 
            
train, test = load_training_data(limit=10000)
train_model(train, test, iterations=10)

def test_model(input_data: str):
    # Загружаем сохраненную модель
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    # Определяем возвращаемое предсказание
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв"
        score = parsed_text.cats["neg"]
    print(f"Текст обзора: {input_data}\n\
          Предсказание: {prediction}\n\
              Score: {score:.3f}")    
              
TEST_REVIEW  = "пришли быстро, но на размер  меньше"         
test_model(input_data=TEST_REVIEW)