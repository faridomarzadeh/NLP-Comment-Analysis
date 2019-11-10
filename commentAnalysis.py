import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import datetime
from collections import Counter
import io

def get_Frequency(word,set):
    freq=0
    if word in set:
        freq=set[word]
    return  freq

def conditional_probability(word,vocabulary_length,category_length,frequency_of_word,smooth):
    return '{:f}'.format((frequency_of_word+smooth)/(category_length+vocabulary_length))

all_df=pd.read_csv('hn2018_2019.csv')
training_df=all_df.loc[pd.to_datetime(all_df['Created At'])<datetime.date(2019,1,1)]
testing_df=all_df.loc[pd.to_datetime(all_df['Created At'])>=datetime.date(2019,1,1)]
lemmatizer = WordNetLemmatizer()
training_df['Title']=training_df['Title'].apply(lambda line:' '.join([lemmatizer.lemmatize(word) for word in line.rstrip().split()]))
vocabulary=[]
for index,row in training_df.iterrows():
    vocabs=list(row['Title'].split(' '))
    vocabulary.extend(vocabs)
frequency=Counter(vocabulary)
print(len(frequency.keys()))
gk=training_df.groupby('Post Type')
story_vocabs=[]
ask_hn_vocabs=[]
show_hn_vocabs=[]
poll_vocabs=[]
story=gk.get_group('story')
ask_hn=gk.get_group('ask_hn')
show_hn=gk.get_group('show_hn')
poll=gk.get_group('poll')
for index,row in story.iterrows():
    story_vocabs.extend(list(row['Title'].split(' ')))
frequency_story=Counter(story_vocabs)

for index,row in ask_hn.iterrows():
    ask_hn_vocabs.extend(list(row['Title'].split(' ')))
frequency_ask_hn=Counter(ask_hn_vocabs)

for index,row in show_hn.iterrows():
    show_hn_vocabs.extend(list(row['Title'].split(' ')))
frequency_show_hn=Counter(show_hn_vocabs)

for index,row in poll.iterrows():
    poll_vocabs.extend(list(row['Title'].split(' ')))
frequency_poll=Counter(poll_vocabs)
filename='model-2018.txt'
file=io.open(filename,'w',encoding='utf-8')
k=0
vocabulary_len=len(frequency)
story_len=len(frequency_story)
ask_hn_len=len(frequency_ask_hn)
show_hn_len=len(frequency_show_hn)
poll_len=len(frequency_poll)
for word in frequency.keys():
    word_in_story_freq=get_Frequency(word,frequency_story)
    story_probability=conditional_probability(word,vocabulary_len,story_len,word_in_story_freq,0.5)
    word_in_ask_hn_freq=get_Frequency(word,frequency_ask_hn)
    ask_hn_probability=conditional_probability(word,vocabulary_len,ask_hn_len,word_in_ask_hn_freq,0.5)
    word_in_show_hn_freq=get_Frequency(word,frequency_show_hn)
    show_hn_probability=conditional_probability(word,vocabulary_len,show_hn_len,word_in_show_hn_freq,0.5)
    word_in_poll_freq=get_Frequency(word,frequency_poll)
    poll_probability=conditional_probability(word,vocabulary_len,poll_len,word_in_poll_freq,0.5)
    file.write(str(k)+'  '+str(word)+'  '+str(word_in_story_freq)+'  '+str(story_probability)+'  '+
               str(word_in_ask_hn_freq)+'  '+str(ask_hn_probability)+'  '+
               str(word_in_show_hn_freq)+'  '+str(show_hn_probability)+'  '+
               str(word_in_poll_freq)+'  '+str(poll_probability))
    file.write('\n')
    k+=1
file.close()
print('Done')
'''
print(len(gk.get_group('story')))
print(len(gk.get_group('ask_hn')))
print(len(gk.get_group('show_hn')))
print(len(gk.get_group('poll')))
'''


