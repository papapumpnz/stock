#!/usr/bin/env python
# -*- coding: utf-8 -*-

train_only=True            # dont download news feeds, use what we have in db and train the data

import sys
import os

# additonal python scripts in this path
pyCustomScripts=['scrapy_modules','modules']
# addtional libraries in this path
pyCustomLibraries=[]
# get correct scrapy settings file
os.environ['SCRAPY_SETTINGS_MODULE'] = 'settings'

# get current script run directory
rootDir=os.getcwd()

# setup additional search paths
for script in pyCustomScripts:
  sys.path.append(rootDir + "/" + script)
for script in pyCustomLibraries:
  if os.environ.has_key('CLASSPATH'):
    if len(os.environ['CLASSPATH'])>0:
      os.environ['CLASSPATH']=os.environ['CLASSPATH'] + ';' + rootDir + script
    else:
      os.environ['CLASSPATH']=os.environ['CLASSPATH'] + rootDir + script
  else:
    os.environ['CLASSPATH']=rootDir + script

import logging
import logging.config

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.conf import settings
from scrapy.utils.log import configure_logging

import importlib
import base64
import re
import database
from datetime import datetime,date,timedelta
import time
from collections import Counter
from pprint import pprint

from yahoo_finance import Share

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from pandas import DataFrame

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np


#---------------------C L A S S E S ----------------------------------- 
class Spiders():

  def __init__(self,logger,process,settings):
      self.spiders = []         # dict of all the spiders
      self.logger=logger        # get logger object
      self.process=process

      # setup proxy authentication details if enabled
      if settings['PROXY_ENABLED']==True:
        logger.info("Proxy server enabled")
        
        # decode password if encoded
        result=re.search('^ENC\((.*)\)',settings['PROXY_PASSWORD'])
        if result:
          proxy_password=base64.b64decode(result.group(1))
        else:
          proxy_password=settings['PROXY_PASSWORD']
          
        proxy='http://%s:%s@%s' % (settings['PROXY_USERNAME'],proxy_password,settings['PROXY_SERVER'])
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
      
  def findSpiders(self,spider_dir):
    # give it a directory full of spider modules

    # setup additional search paths
    sys.path.append(rootDir + '/' + spider_dir)
      
    for name in os.listdir(spider_dir):
      if name.endswith(".py"):
        #strip the extension
        module = name[:-3]
        # set the module name in the current global name space:
        try:
          spider_class=getattr(importlib.import_module(module), "mySpider")
        except ImportError as e:
          self.logger.error("Unable to import spider module %s in path %s. Error %s" % (module,rootDir + '/' +  spider_dir,e))
          return

        self.addSpider(spider_class())

  def addSpider(self,spider):
    # adds spider to crawl list
    self.process.crawl(spider)
    self.spiders.append(spider)
    #self.process.signals.connect(self.spiderClosed, signal=signals.spider_closed)
    self.logger.info("Added %s spider to crawl list" % spider.name)
  
  def getUserAgent(self):
    # returns the default user agent
    return self.user_agent
    
  def listSpiders(self):
    # prints a list of spider modules loaded
    if len(self.spiders) == 0 :
      self.logger.info("  None")
    else:
      for spider_name in self.spiders:
        self.logger.info("  %s" % spider_name.name)
      
  def getSpiders(self):
    # returns a list of spider modules loaded
    return self.spiders
      
  def startAllSpiders(self):
    # starts crawl of all spiders
    self.process.start() # the script will block here until the crawling is finished


#---------------------M E T H O D S -----------------------------------

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        #print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    filename='test_plot_classif_report_%s.png' % title
    plt.savefig(filename, dpi=200, format='png', bbox_inches='tight')
    plt.close()

def get_avg_precision(cr):
    # returns the avg precesion of a classification report
    lines = cr.split('\n')
    avg_line=lines[6 : (len(lines) - 1)]
    line=str(avg_line).split(' ')
    return(str(line[9]))

 
#-------------------- M A I N -----------------------------------------
def main():
    """
    The main entry point of the application
    """
    
    # record start time
    script_start = time.time()
    
    logging.config.fileConfig('log.conf')
    logger = logging.getLogger(__name__)

    # get scrapy logging
    configure_logging(install_root_handler=False)

    # print the syspath
    logger.info("Python sys.path variable is : %s" % sys.path)
    if os.environ.has_key('CLASSPATH'):
        logger.info("Python class.path variable is : %s" % os.environ['CLASSPATH'])

    logger.info("Program started")

    # create a scrapy crawler process
    process = CrawlerProcess(settings)

    # create a spider object
    spider=Spiders(logger,process,settings)

    # connect to the trainer database
    connection=database.connect(settings['TRAINER_MONGODB_SERVER'],settings['TRAINER_MONGODB_PORT'],settings['TRAINER_MONGODB_DB'],settings['TRAINER_MONGODB_COLLECTION'],settings['TRAINER_MONGODB_UNIQ_KEY'])
    

    if not train_only:
        ###
        #   DATAFEEDS CRAWL
        ###  
        logger.info("Getting news data feed")
        
        # load in company info spiders
        spider.findSpiders('datafeeds') 

        # list spiders
        logger.info("Spiders loaded are:")
        spider.listSpiders()

        # run all spiders
        spider.startAllSpiders() # the script will block here until the last crawl call is finished


        ###
        #   CLASSIFY NEWS ITEMS
        ###
        logger.info("Classfying news")

        # get urls that have no classification
        db_records=database.get_all(connection,{'classification':None},True)

        if db_records:
            logger.debug("Found %i records to classify" % int(db_records.count()))
            for record in db_records:
                
                # pause execution
                time.sleep(settings['YAHOO_FINANCE_DOWNLOAD_DELAY'])
                
                logger.info("Classifying news item %s" % record['url'])
                
                item={}
                
                # check if news item is older than CLASSIFICATON_POST_NEWS_OLDER_THAN_DAYS
                item_date=datetime.strptime(record['date'],'%Y-%m-%d').date()
                now_date=date.today()
                check_date=now_date-timedelta(days=int(settings['CLASSIFICATON_POST_NEWS_OLDER_THAN_DAYS']))
                #print("item_date : %s" %str(item_date))
                #print("check_date : %s" %str(check_date))
                if (item_date <= check_date):
                
                    # get historical data for this news item
                    from_date=datetime.strptime(record['date'],'%Y-%m-%d').date()
                    to_date=from_date+timedelta(days=settings['PRICE_HISTORY_DAYS'])
                    logger.info('Retrieving price history from %s to %s' % (str(from_date),str(to_date)))        
                    ticker_name='%s.%s' % (record['ticker'],record['se'])
                    try:
                        yahoo = Share(ticker_name)
                        item['price_history']=yahoo.get_historical(str(from_date),str(to_date))
                        item['price_200_avg']=yahoo.get_200day_moving_avg()
                    except Exception as e:
                        logger.warning("Error retrieving historical share price for ticker %s. Error was : %s" %(ticker_name,e))
                        item['price_history']=None
                            
                    if item['price_history']:
                        # calculate classification
                        # take first last record (since its actually the date of the news item) and take the first (later date) and calculate % diff
                        his_len=(len(item['price_history'])-1)
                        if 'Open' in item['price_history'][his_len]:
                            start=item['price_history'][his_len]['Open']
                        else:
                            logger.warning('Incomplete price history : %s' % item['price_history'][his_len])
                            start=None
                            
                        if 'Close' in item['price_history'][0]:
                            end=item['price_history'][0]['Close']
                        else:
                            logger.warning('Incomplete price history : %s' % item['price_history'][0])
                            end=None
                        
                        # check if we have a start and end price, if not skip this
                        if start and end:
                            post_news_gain=(float(end)-float(start))/float(start)    # 3 day close price - start price / start price
                            item['percent_gain']=str(post_news_gain)
                            item['3_day_close']=end
                            
                            # get volume of trades over that period. We want to see trade volume movement otherwise the news item had no impact on price
                            trade_vol=0
                            for price in item['price_history']:
                                trade_vol=trade_vol+int(price['Volume'])
                            
                            # determine if news item caused gain or loss
                            if trade_vol<=settings['CLASSIFICATON_POST_NEWS_TRADE_VOL_GRT']:
                                item['classification']='na'
                            else:
                                if post_news_gain<0:
                                    item['classification']='neg'
                                else:
                                    item['classification']='pos'
                            
                            # write the record back
                            #print(item)
                            result=database.put_one(connection,{settings['TRAINER_MONGODB_UNIQ_KEY']:record['url']},item)
                        else:
                            # pass start and end check
                            pass
                    else:
                        # pass if item price history
                        pass
                else:
                    # pass if news item newer than CLASSIFICATON_POST_NEWS_OLDER_THAN_DAYS
                    logger.info("News item %s is newer than %s days. Skipping." % (record['url'],settings['CLASSIFICATON_POST_NEWS_OLDER_THAN_DAYS']))
                    pass
        else:
            # no records to classify
            logger.debug("Found 0 records to classify")
            
        # close mongo cursor
        db_records.close()
        
        
    ###
    #   Train
    ###
    logger.info("Training")
    
    # get all our train data
    logger.debug("  Getting data from database")
    db_records=database.get_all(connection,{'classification':{'$ne':None}})
    
    # get our train data into a pandas dataframe
    logger.debug("  Putting data into dataframe")

    rows=[]
    index=[]
    
    for record in db_records:
        rows.append({'text':record['text'], 'class':record['classification']})
        index.append(record['url'])
    
    data = DataFrame({'text': [], 'class': []})
    data = DataFrame(rows, index=index)
    data = data.reindex(np.random.permutation(data.index))
    

    
    # check dataframe content
    #print ("Head")
    #print (data.head(10))
    
    # check our dataframe columns
    print ("\nColumns check")
    print ("  %s" % data.columns)
    # print elements
    print("  The data-set has %d rows and %d columns"%(data.shape[0],data.shape[1]))
    
    print("\nMissing data check")
    for col_name in data.columns:
        print ("  %s : %s" % (col_name,sum(data[col_name].isnull())))
    
    #print("\nClass distribution pre")
    #print(data.describe(include='all'))
    
    print("\nDuplicates check")
    print(sum(data.duplicated()))
    
    print("\nCategory count")
    category_counter={x:0 for x in set(data['class'])}
    for each_cat in data['class']:
        category_counter[each_cat]+=1
    print(category_counter)
    
    # format text
    print("\nTransforming data")
    corpus=data.text
    all_words=[w.split() for w in corpus]
    all_flat_words=[ewords for words in all_words for ewords in words]
    
    #removing all the stop words from the corpus
    all_flat_words_ns=[w for w in all_flat_words if w not in stopwords.words("english")]
    
    #removing all duplicates
    set_nf=set(all_flat_words_ns)

    print("Number of unique vocabulary words in the text column of the dataframe: %d"%len(set_nf))
    
    porter=nltk.PorterStemmer()
    for each_row in data.itertuples():
        m1=map(lambda x: x,(each_row[2]).lower().split())
        #for each row converts them to lower case.
        m2=[]
        for word in m1:
            m2.append(''.join(e for e in word if e.isalpha()))
        #for each row removes words with digits
        m3=map(lambda x: porter.stem(x),m2)
        #Using Porter Stemmer in NLTK, stemming is performed on the str created in previous step.
        data.loc[each_row[0],'text_proc']=' '.join(m3)

    print("\nClass distribution post")
    print(data.describe(include='all'))

    print ("\nHead")
    print (data.head(10))
    
    corpus=data.text_proc
    
    pipeline = Pipeline([
		('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
		('classifier',       MultinomialNB())
    ])   

    # train the data
    logger.debug("  training data")
   
    k_fold = KFold(n=len(data), n_folds=2)
    
    scores = []
    confusion = np.array([[0, 0,0], [0, 0,0], [0, 0,0]])
    for train_indices, test_indices in k_fold:
        train_text = data.iloc[train_indices]['text_proc'].values
        train_y = data.iloc[train_indices]['class'].values.astype(str)

        test_text = data.iloc[test_indices]['text_proc'].values
        test_y = data.iloc[test_indices]['class'].values.astype(str)
        
        pipeline.fit(train_text, train_y)
        predictions = pipeline.predict(test_text)
        
        confusion += confusion_matrix(test_y, predictions)
        score = f1_score(test_y, predictions,average=None)
        scores.append(score)

    print('Total classified:', len(data))
    print('Score:', sum(scores)/len(scores))
    print('Confusion matrix:')
    print(confusion)    
    

    

    ###
    #   Finish
    ###
    logger.info("Saving prediction model")
    joblib.dump(pipeline, settings['PRED_FILE_NAME'])

    # record end time
    script_end = time.time()
    script_run_time = script_end-script_start

    # dump stats data to database
    logger.info("Writing trainer statistics to database")
    
    stats={}
    
    # connect to the trainer database
    stats_connection=database.connect(settings['STATS_MONGODB_SERVER'],settings['STATS_MONGODB_PORT'],settings['STATS_MONGODB_DB'],settings['STATS_MONGODB_COLLECTION'],settings['STATS_MONGODB_UNIQ_KEY'])
    
    # construct the stats
    #stats['precesion']=get_avg_precision(report_lin)
    #stats['train_time']=time_linear_train
    #stats['predict_time']=time_linear_predict
    #stats['pos_labels']=tr_data_count['pos']
    #stats['neg_labels']=tr_data_count['neg']
    #stats['na_labels']=tr_data_count['na']
    #stats['run_time']=str(script_run_time)
    
    # write the stats (no index here so use None)
    #database.put_one(stats_connection,None,stats)

  
    # finish
    logger.info("All spiders finished. Run time was %s seconds" % script_run_time)
 
if __name__ == "__main__":
    main()