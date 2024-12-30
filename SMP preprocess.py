import numpy as np
import pandas as pd
import collections
import jieba
import time
import os
import re

raw_train_data_url = "https://worksheets.codalab.org/rest/bundles/0x0161fd2fb40d4dd48541c2643d04b0b8/contents/blob/"
raw_test_data_url = "https://worksheets.codalab.org/rest/bundles/0x1f96bc12222641209ad057e762910252/contents/blob/"

def get_json_data(path):
    data_df = pd.read_json(path)
    data_df = data_df.transpose()
    cols = data_df.columns.tolist()
    data_df = data_df[['query', 'label']]
    text_index = cols.index('query')
    label_index = cols.index('label')
    cols[text_index], cols[label_index] = cols[label_index], cols[text_index]
    data_df = data_df[cols].rename(columns={'query': 'text'})
    return data_df

def chinese_pre(text_data):
    text_data = text_data.lower() #换成小写
    #分词
    text_data = list(jieba.cut(text_data,cut_all=False))
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    text_data = " ".join(text_data)
    return text_data

if (not os.path.exists('./data/train.json')) or (not os.path.exists('./data/dev.json')):
    raw_train = requests.get(raw_train_data_url)
    raw_test = requests.get(raw_test_data_url)
    if not os.path.exists('./data'):
        os.makedirs('./data')
    with open("./data/train.json", "wb") as code:
         code.write(raw_train.content)
    with open("./data/dev.json", "wb") as code:
         code.write(raw_test.content)

if __name__ == '__main__':

    preprocess_path = os.path.join(os.getcwd(),'preprocessed data')
    if not os.path.exists(preprocess_path):
        os.mkdir(preprocess_path)

    else: pre = True
    pre = False
    if not pre:
        label = {'website':"网站", 'tvchannel':"电视", 'lottery':"彩票", 'chat':"聊天", 'match':"比赛",
                  'datetime':"日期", 'weather':"天气", 'bus':"汽车", 'novel':"小说", 'video':"音频", 'riddle':"谜语",
                  'calc':"计算", 'telephone':"电话", 'health':"健康", 'contacts':"交流", 'epg':"未知", 'app':"应用", 'music':"音乐",
                  'cookbook':"食谱", 'stock':"股票", 'map':"地图", 'message':"信息", 'poetry':"诗歌", 'cinemas':"影院", 'news':"新闻",
                  'flight':"航班", 'translation':"翻译", 'train':"训练", 'schedule':"安排", 'radio':"音响", 'email':"邮件"}
        print(len(label))
        train_data_df = get_json_data(path="data/train.json")
        test_data_df = get_json_data(path="data/dev.json")

        train_data_df["labels"] = train_data_df["label"].map(label)
        test_data_df["labels"] = test_data_df["label"].map(label)

        train_data = list(zip(train_data_df["labels"],train_data_df["text"]))
        test_data = list(zip(test_data_df["labels"], test_data_df["text"]))


        with open(os.path.join(preprocess_path,"train data.txt"),'w',encoding='utf-8') as file:
            for pair in train_data:
                file.write(pair[0] + '\t' + pair[1] + '\n')

        with open(os.path.join(preprocess_path,"test data.txt"),'w',encoding='utf-8') as file:
            for pair in test_data:
                file.write(pair[0] + '\t' + pair[1] + '\n')
# if __name__ == '__main__':
#
#     preprocess_path = os.path.join(os.getcwd(),'preprocessed data')
#     if not os.path.exists(preprocess_path):
#         os.mkdir(preprocess_path)
#         pre = False
#     else: pre = True
#
#     if not pre:
#         label = ['website', 'tvchannel', 'lottery', 'chat', 'match',
#                   'datetime', 'weather', 'bus', 'novel', 'video', 'riddle',
#                   'calc', 'telephone', 'health', 'contacts', 'epg', 'app', 'music',
#                   'cookbook', 'stock', 'map', 'message', 'poetry', 'cinemas', 'news',
#                   'flight', 'translation', 'train', 'schedule', 'radio', 'email']
#         labels = {}
#         for idx, lab in enumerate(label):
#             labels[lab] = idx
#         train_data_df = get_json_data(path="data/train.json")
#         test_data_df = get_json_data(path="data/dev.json")
#         stop_words = pd.read_csv("./cn_stopwords.txt",sep="\t",header=None,names=["text"])
#
#         train_data_df["cutword"] = train_data_df['text'].apply(chinese_pre)
#         test_data_df["cutword"] = test_data_df['text'].apply(chinese_pre)
#
#         train_data_df["labelcode"] = train_data_df["label"].map(labels)
#         test_data_df["labelcode"] = test_data_df["label"].map(labels)


