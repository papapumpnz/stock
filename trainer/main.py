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

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

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
 
#-------------------- M A I N -----------------------------------------
def main():
    """
    The main entry point of the application
    """
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
        db_records=database.get_all(connection,{'classification':None})

        if db_records:
            logger.debug("Found %i records to classify" % int(db_records.count()))
            for record in db_records:
                
                logger.info("Classifying news item %s" % record['url'])
                
                item={}
                
                # get historical data for this news item
                from_date=datetime.strptime(record['date'],'%Y-%m-%d').date()
                to_date=from_date+timedelta(days=settings['PRICE_HISTORY_DAYS'])
                logger.info('Retrieving price history from %s to %s' % (str(from_date),str(to_date)))        
                ticker_name='%s.%s' % (record['ticker'],record['se'])
                try:
                    yahoo = Share(ticker_name)
                    item['price_history']=yahoo.get_historical(str(from_date),str(to_date))
                except Exception as e:
                    logger.warning("Error retrieving historical share price for ticker %s. Error was : %s" %(ticker_name,e))
                    item['price_history']=None
                        
                if item['price_history']:
                    # calculate classification
                    # take first last record (since its actually the date of the news item) and take the first (later date) and calculate % diff
                    his_len=(len(item['price_history'])-1)
                    start=item['price_history'][his_len]['Open']
                    end=item['price_history'][0]['Open']
                    int_diff=(float(end)-float(start))
                    
                    # get volume of trades over that period. We want to see trade volume movement otherwise the news item had no impact on price
                    trade_vol=0
                    for price in item['price_history']:
                        trade_vol=trade_vol+int(price['Volume'])
                    
                    if trade_vol<=settings['CLASSIFICATON_POST_NEWS_TRADE_VOL_GRT']:
                        item['classification']='na'
                    else:
                        if int_diff<0:
                            item['classification']='neg'
                        else:
                            item['classification']='pos'
                    
                    # write the record back
                    #print(item)
                    result=database.put_one(connection,{settings['TRAINER_MONGODB_UNIQ_KEY']:record['url']},item)
                else:
                    pass
        else:
            logger.debug("Found 0 records to classify")
        
        
    ###
    #   PROCESS NEWS TEXT - TOKENIZE, LOWERCASE, LEMMATIZE and STOP WORD PROCESSING
    ###
    logger.info("Normalising news text")

    # get urls that have not had text processing
    db_records=database.get_all(connection,{'text_processed':None})

    if db_records:
        logger.debug("Found %i records to process for news text" % int(db_records.count()))
        for record in db_records:
            
            logger.info("Converting news text for item %s" % record['url'])
            
            item={}
            
            # conver to lowercase
            the_text=record['text'].lower()
            
            # tokenize
            tokens = word_tokenize(the_text)
            
            # remove stop words
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            
            # lemmatize
            wnl = nltk.WordNetLemmatizer()
            item['text']=[wnl.lemmatize(t) for t in tokens]
            item['text_processed']='true'
            
            # write the record back
            #print(item)
            result=database.put_one(connection,{settings['TRAINER_MONGODB_UNIQ_KEY']:record['url']},item)
            

    else:
        logger.debug("Found 0 records to process for news text")    

    ###
    #   Train
    ###
    logger.info("Training")
    
    # get all our train data
    logger.debug("  Getting data from database")
    db_records=database.get_all(connection,{'classification':{'$ne':None}})
    
    # format our train data into an array
    logger.debug("  Formatting data")
    tr_data=[]
    tr_labels=[]
    
    for record in db_records:
        tr_labels.append(record['classification'])
        tr_data.append(record['text'])
        
    
    #pprint(tr_data)
    
    count = Counter(tr_labels)
    logger.debug("    Label count : %s" %count.most_common(10))
    
    # split our data into a 20% test and an 80% train data sets
    logger.debug("  Splitting datasets")
    train_data,test_data,train_labels,test_labels=train_test_split(tr_data,tr_labels, test_size=0.2, random_state=45)
    
    logger.debug("    Train data : %s" % len(train_data))
    #logger.debug("Train labels : %s" % len(train_labels))
    logger.debug("    Test data : %s" % len(test_data))
    #logger.debug("Test labels : %s" % len(test_labels))
    
    # Create feature vectors
    logger.debug("  Creating feature vectors")
    tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    train_vectors = tfidf.fit_transform(train_data)
    test_vectors = tfidf.transform(test_data)
    
    # Perform classification with SVM, kernel=rbf
    logger.debug("  Preforming classification with SVM")
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1
     
    logger.info("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
  
    # finish
    logger.info("All spiders finished")
 
if __name__ == "__main__":
    main()