# from __future__ import print_function

import argparse

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca import OrcaContext
import os
# recommended to set it to True when running Analytics Zoo in Jupyter notebook. 
OrcaContext.log_output = False # (this will display terminal's stdout and stderr in the Jupyter notebook).

cluster_mode = "local"

if cluster_mode == "local":
    init_orca_context(cores=1, memory="14g") # run in local mode
elif cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4) # run on K8s cluster
elif cluster_mode == "yarn":
    init_orca_context(
        cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
        driver_memory="10g", driver_cores=1
        ) # run on Hadoop YARN cluster


import numpy as np
from zoo.tfpark import TFDataset

import os
import argparse
from pyspark.context import SparkContext 
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import col, when
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, RegexTokenizer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, Word2Vec
from pyspark.sql import Row
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
import os
import glob
import pandas as pd
import numpy as np
# recommended to set it to True when running Analytics Zoo in Jupyter notebook. 

from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler

import re
from emoji import get_emoji_regexp
import unicodedata
from underthesea import word_tokenize

from bigdl.dlframes.dl_classifier import *
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.types import *
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.dlframes.dl_classifier import *
from pyspark.sql.types import *
from pyspark.sql.functions import col,length,trim
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.feature.common import *

def load_dataset(path):
    # import ast
    # folder_name = ast.literal_eval(folder)
    # address_files = []
    # for i in folder_name:
    #   address_files += glob.glob(path+i+'/*.xlsx')
    # print(address_files)
    data = pd.read_csv(path)
    # sum_read_file=[]
    # for i in address_files:
    #   temp_file = pd.read_excel(i)
    #   temp_file.columns = temp_column
    
    #   sum_read_file.append(temp_file)
    # sum_read_file = pd.concat(sum_read_file)
    sparkDF=spark.createDataFrame(data.astype(str))
    return sparkDF


class Preporcessing:
  def __init__(self, advance_preprocessing = False, embedding_type = "tfidf", path_acronyms = None):
    self.advance_prep = advance_preprocessing
    self.embedding_type = embedding_type
    if path_acronyms:
      self.dict_special = self.special_case(path_acronyms)
  
  def special_case(self, path_acronyms):
      special_w = pd.read_excel(path_acronyms)
      special_w = special_w.to_dict('index')
      dict_special={}
      for key, values in special_w.items():
        row = []
        for k,v in values.items():
          if len(v)>=3:
            row.append(v)
        if len(row) ==2:
          dict_special.update({row[1]:[row[0]]})
      return dict_special

  def clean_text(self, text, special_w=None):
    # Unicode normalize
    text = unicodedata.normalize('NFC',text)

    # Lower
    text = text.lower()

    # Remove all emoji
    text = re.sub(get_emoji_regexp(),"",text)

    #  Change #string to HASTAG 
    if self.advance_prep:
        text = re.sub('#\S+',"HASTAG",text)

        # # Find all price tag and change to GIÁ
        pricetag = '((?:(?:\d+[,\.]?)+) ?(?:nghìn đồng|đồng|k|vnd|d|đ))'
        text = re.sub(pricetag,"PRICE",text)

        # Replace some special word
        replace_dct = {"òa ":["oà "], "óa ":["oá "], "ỏa ":["oả "], "õa ":["oã "], "ọa ":["oạ "],
                  "òe":["oè"], "óe":["oé"], "ỏe":["oẻ"], "õe":["oẽ"], "ọe":["oẹ"],
                  "ùy":["uỳ"], "úy":["uý"], "ủy":["uỷ"], "ũy":["uỹ"], "ụy":["uỵ"],
                  "ùa":["uà"], "úa ":["uá "], "ủa":["uả"], "ũa":["uã"], "ụa":["uạ"],
                  "xảy":["xẩy"], "bảy":["bẩy"], "gãy":["gẫy"],"nhân viên ":["nvien"],"quay":['qay']}
        sum_special =  {**special_w, **replace_dct}    
        for key, values in sum_special.items():
          if type(values) == list:
            for v in values:
              text = text.replace(v, key)
        text = text.replace('ìnhh','ình')

    # Remove all special char
    specialchar = r"[\"#$%&'()*+,\-\/\\:;<=>@[\]^_`{|}~\n\r\t]"
    text = re.sub(specialchar," ",text)

    if self.advance_prep:
        text = word_tokenize(text, format="text")

    return text

  def clean_df(self, sparkDF):
    Clean_UDF = udf(lambda x: self.clean_text(x,self.dict_special),StringType())
    # Clean_Nan = udf (lambda x: label_encode[2] if x=='nan' else label_encode[int(float(x))],ArrayType(StringType()))
    Clean_Nan = udf (lambda x: float(-2.0) if x not in ['0.0','1.0','-1.0'] else float(x), FloatType())
    DF_Clean = sparkDF.select(Clean_UDF('cmt').alias("cmt") , Clean_Nan('general').alias("general"), Clean_Nan('price').alias("price"), Clean_Nan('quality').alias("quality"), Clean_Nan('service').alias("service"), Clean_Nan('stylefood').alias("stylefood"),Clean_Nan('location').alias("location"), Clean_Nan('background').alias("background"))
    return DF_Clean.withColumn("label", f.array("general",'price',"quality","service","stylefood","location","background").cast(ArrayType(FloatType())))
  
  def clean_sentenceDF(self, sentenceDF):
    Clean_UDF = udf(lambda x: self.clean_text(x,self.dict_special),StringType())
    DF_Clean = sentenceDF.select(Clean_UDF('cmt').alias("cmt"))
    return DF_Clean

  def split_data(self, sparkDF, train_ratio = 0.8, seed = 50):
    train_data, test_data = sparkDF.randomSplit([train_ratio, 1-train_ratio], seed)
    return train_data, test_data

  def Embedding(self, num_feature = None):
    tokenizer = Tokenizer(inputCol="cmt", outputCol="words")
    newdb = VectorAssembler(inputCols=["features_vec"], outputCol="features")

    if self.embedding_type == "wordcount":
      if num_feature == None:
          num_feature = 3000
      countVectors = CountVectorizer(inputCol="words", outputCol="features_vec", minDF=5, vocabSize=num_feature)
      pipeline = Pipeline(stages=[tokenizer,countVectors,newdb])

    elif self.embedding_type == "tfidf":
      if num_feature == None:
          num_feature = 10000
      hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=num_feature)
      idf = IDF(inputCol="rawFeatures", outputCol="features_vec" )
      pipeline = Pipeline(stages=[tokenizer,hashingTF,idf,newdb])

    elif self.embedding_type == "word2vec":
      if num_feature == None:
          num_feature = 300
      w2v = Word2Vec(vectorSize=num_feature, seed=42, inputCol="words", outputCol="features_vec")
      pipeline = Pipeline(stages=[tokenizer, w2v,newdb])
      
    else:
      raise ValueError("Embedding phải là 'wordcount', 'tfidf' hoặc 'word2vec'. Các embedding khác chưa hỗ trợ.")
    
    return pipeline

def convertCase(float_num):
  """ so sánh với 0,-1,1,-2""" 
  value_with_nan = abs(-2-float_num)
  value_with_neu = abs(0-float_num)
  value_with_neg = abs(-1-float_num)
  value_with_pos = abs(1-float_num)
  value_min = min([value_with_nan,value_with_neu,value_with_neg,value_with_pos])
  if value_min == value_with_nan:
    return -2.
  elif value_min== value_with_neu:
    return 0.
  elif value_min == value_with_neg:
    return -1.
  return 1.

def edit_prediction_label(prediction_data):
  edit_pred_label = udf(lambda label_list: [convertCase(x) for x in label_list],ArrayType(FloatType()))
  list_pre = prediction_data.withColumn("prediction",edit_pred_label('prediction'))
  return list_pre

class Model:
  def __init__(self, model, input_dim, batch_size = 64,learning_rate = 0.2,epoch = 10,criterion=MSECriterion()):
    self.criterion=criterion
    self.model = model
    self.est =  NNEstimator(model, criterion, SeqToTensor([input_dim]), ArrayToTensor([7])) \
                .setBatchSize(batch_size).setLearningRate(learning_rate).setMaxEpoch(epoch) 
    self.NNmodel = None
  
  def train(self,train_data):
    self.NNmodel = self.est.fit(train_data)

  def predict(self,test_data):
    return self.NNmodel.transform(test_data)

  def evaluate(self, predict_data):
    res_final = edit_prediction_label(predict_data)
    list_pre = res_final.select('label','prediction').collect()
    acc_aspect = [0,0,0,0,0,0,0]
    total = len(list_pre)
    for i in range(0,len(list_pre)):
      for j in range(7):
        if list_pre[i][0][j] == list_pre[i][1][j]:
          acc_aspect[j] +=1
    return acc_aspect

  def save_model(self, path_model):
    self.NNmodel.save(f"{path_model}")
    print("Save nnmodel successfull!")
  
  def load_model(self, path):
    from zoo.pipeline.nnframes.nn_classifier import NNModel
    self.NNmodel = NNModel(self.model)
    self.NNmodel = self.NNmodel.load(path)
    print("Load nnmodel successfull!")

def LSTM_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(LSTM(input_size, hidden_size))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Dropout(0.2))
    model.add(Linear(hidden_size, output_size))
    return model

def GRU_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(GRU(input_size, hidden_size))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Dropout(0.2))
    model.add(Linear(hidden_size, output_size))
    return model

# def LogisticRegression(input_size, output_size):
#     model = Sequential()
#     model.add(Linear(input_size, output_size))
#     model.add(Sigmoid())
#     return model

def MLP(input_size, hidden_size, hidden_size2, output_size):
    model = Sequential()
    model.add(Linear(input_size, 1000))
    model.add(ReLU())
    model.add(Linear(1000, 256))
    model.add(ReLU())
    model.add(Linear(256, output_size))
    return model

def RNN_model(input_size, hidden_size, output_size):
    model = Sequential()
    recurrent = Recurrent()
    recurrent.add(RnnCell(input_size, hidden_size, Tanh()))
    model.add(InferReshape([-1, input_size], True))
    model.add(recurrent)
    model.add(Select(2, -1))
    model.add(Dropout(0.2))
    model.add(Linear(hidden_size, output_size))
    return model

#thu vien
import streamlit as st
import SessionState
import pandas as pd
ss = SessionState.get(lst = [])

def prepare_model(model_type, embedding_type, advance_prep, model_path, acronyms_path):
    if advance_prep:
      prep_name = ""
    else:
      prep_name = "-basicprepro"
    embedding_path = model_path + "embedding_model"
    if embedding_type == "TF-IDF":
        embedtype = 'tfidf'
        embedding_path += "/tfidf" + prep_name
        embedding_dim = 10000
    elif embedding_type =="Count Vectorizer":
        embedtype = "wordcount"
        embedding_path += "/wordcount" + prep_name
        embedding_dim = 3000
    elif embedding_type =="Word2Vec":
        embedtype = "word2vec"
        embedding_path += "/word2vec" + prep_name
        embedding_dim = 300

    if model_type == "LSTM":
        model_path = model_path + "LSTM/lstm-" +  embedtype + prep_name
        model = LSTM_model(embedding_dim, 256, 7)
    elif model_type == "GRU":
        model_path = model_path + "GRU/gru-" +  embedtype + prep_name
        model = GRU_model(embedding_dim, 256, 7)
    elif model_type == "Multi-layer Perceptron (MLP)":
        if embedtype == "word2vec":
          hidden_size = 256
          hidden_size2 = 128
        else:
          hidden_size = 1000
          hidden_size2 = 256 
        model_path = model_path + "MLP/mlp-" +  embedtype + prep_name
        model = MLP(embedding_dim, hidden_size, hidden_size2, 7)
    elif model_type == "RNN":
        model_path = model_path + "RNN/rnn-" +  embedtype + prep_name
        model = RNN_model(embedding_dim, 256, 7)

    print(embedding_path)
    print(model_path)
    preprocessing = Preporcessing(advance_preprocessing=advance_prep, embedding_type=embedding_type, path_acronyms=acronyms_path)
    print("Load embedding ... ")
    embedding = PipelineModel.load(embedding_path) 
    print("Load model ...")
    model = Model(model, embedding_dim)
    model.load_model(model_path)
    return model, preprocessing, embedding


def predict(model, preprocessing, embedding, sentence):
    sentence_DF = spark.sql("""select '{}' as cmt""".format(sentence))

    print("Start to Preprocessing...") 
    sentence_DF_cleaned = preprocessing.clean_sentenceDF(sentence_DF)
    sentence_DF_embed = embedding.transform(sentence_DF_cleaned)

    aspect_name = ['general', 'price', 'quality', 'service', 'stylefood','location', 'background']

    sentence_pred = model.predict(sentence_DF_embed)
    sentence_pred = edit_prediction_label(sentence_pred)
    sentence_pred_final = sentence_pred.select([col("prediction")[i].alias(aspect_name[i]) for i in range(7)])

    fill_nan = udf(lambda x: x if x!= -2.0 else '')
    sentence_pred_final = sentence_pred_final.select(fill_nan('general').alias("general"), fill_nan('price').alias("price"), fill_nan('quality').alias("quality"), \
                          fill_nan('service').alias("service"), fill_nan('stylefood').alias("stylefood"),fill_nan('location').alias("location"), fill_nan('background').alias("background")).collect()
    print("Original text: ",sentence)
    print("Predicting:")
    return [sentence_pred_final[0][i] for i in aspect_name]

if __name__ == "__main__":

    spark = OrcaContext.get_spark_session()
    sc = OrcaContext.get_spark_context()

    model_path = "/content/drive/MyDrive/BigData/models/"
    acronyms_path = "/content/drive/MyDrive/BigData/data/original/Acronyms.xlsx"

    # title 
    st.title('ASPECT-BASE SENTIMENT ANALYSIS')
    st.caption("Nhóm 10 - Trần Việt Anh, Lê Thị Thu Hằng, Hoàng Nguyên Phương, Nguyễn Hoàng Bảo Long")

    #dien Form
    with st.form("my_form"):
        # chon model

        st.header('Choose model')
        embedding_type = st.selectbox("Chọn loại embedding",("TF-IDF","Count Vectorizer","Word2Vec"))

        model_type = st.selectbox(
        'Chọn model để phân loại',
        ('LSTM', 'GRU', 'Multi-layer Perceptron (MLP)', 'RNN'))

        advance_prep = st.checkbox("Advanced preprocessing")        
        confirm_model = st.form_submit_button("Confirm")
        if confirm_model:
            ss.lst = []
            model, preprocessing, embedding = prepare_model(model_type, embedding_type, advance_prep, model_path, acronyms_path)
            ss.lst.append(model)
            ss.lst.append(preprocessing)
            ss.lst.append(embedding)
            st.success("Load model sucessfully!")
            print("Load model sucessfully")

        st.header('Prediction')
        # st.write('You selected:', option)
        #nhap comment
        sentence = st.text_area('Nhập comment cần đánh giá:', height = 100)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Predict")
        if submitted:
            ## code hanh dong
            st.write('Predicting ... ')
            try:
              pred = predict(ss.lst[0], ss.lst[1], ss.lst[2], sentence)
              d = {'General': [pred[0]], 'Price': [pred[1]],'Quality': [pred[2]],'Service': [pred[3]],'StyleFood': [pred[4]],'Location': [pred[5]],'Background': [pred[6]]}
              df = pd.DataFrame(d)
              st.table(df)
            except:
              st.info("Please reload your model.")
            
    
    

