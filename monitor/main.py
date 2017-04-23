#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# additonal python scripts in this path
pyCustomScripts=['modules']
# addtional libraries in this path
pyCustomLibraries=[]

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
import settings
import database
import base64
import re
from yahoo_finance import Share
from datetime import datetime

#---------------------C L A S S E S ----------------------------------- 

 
#-------------------- M A I N -----------------------------------------
def main():
  """
  The main entry point of the application
  """
  logging.config.fileConfig('log.conf')
  logger = logging.getLogger(__name__)
  
  tickers=[]
  
  ###
  # GET OUR TICKER DATA
  ###
  
  if settings.database_enabled==True:
    # connect to the ticker database
    logger.info('Attempting to get tickers from database')
    ticker_db=database.connect(settings.ticker_db_server,settings.ticker_db_port,settings.ticker_db_database,settings.ticker_db_collection,settings.ticker_db_uniq_key)
    tickers=database.get_one(ticker_db)
    #logger.info('Retrieved %s tickers' % len(tickers))
    
    # connect to the prices database
    price_db=database.connect(settings.price_db_server,settings.price_db_port,settings.price_db_database,settings.price_db_collection,settings.price_db_uniq_key)
  else:
    # read in an input file
    logger.info('Attempting to get tickers from csv file')
    filename = 'tickers.csv'
    with open(filename, 'r') as f:
      tickers=f.read()
    
    logger.info('Retrieved %s tickers' % len(tickers))

  # setup proxy authentication details if enabled
  if settings.proxy_enabled==True:
    logger.info("Proxy server enabled")
    
    # decode password if encoded
    result=re.search('^ENC\((.*)\)',settings.proxy_password)
    if result:
      proxy_password=base64.b64decode(result.group(1))
    else:
      proxy_password=settings.proxy_password
      
    proxy='http://%s:%s@%s' % (settings.proxy_username,proxy_password,settings.proxy_server)
    os.environ["http_proxy"] = proxy
    os.environ["https_proxy"] = proxy
  
  if settings.database_enabled==False:
    # setup output filename for development only
    filename = 'stock-price.csv'
    file_header="Ticker,Price,50 DMA,50 %,200 DMA, 200 %"
    with open(filename, 'wb') as f:
        f.write(file_header)
        f.write('\n')

  ###
  # PROCESS THE TICKERS AND GET YAHOO STOCK INFORMATION
  ###
  
  logger.info("Processing tickers...")
  for ticker in tickers:
    logger.info("  %s" % ticker['ticker'])
    ticker_name="%s.%s" % (ticker['ticker'],ticker['se'])
    
    try:
      yahoo = Share(ticker_name)
      pr=yahoo.get_price()
      pr1=yahoo.get_50day_moving_avg()
      pr2=yahoo.get_percent_change_from_50_day_moving_average()
      pr3=yahoo.get_200day_moving_avg()
      pr4=yahoo.get_percent_change_from_200_day_moving_average()
    except Exception as e:
      logger.info("Exception getting price for ticker %s. Error:%s" % (ticker['ticker'],e))
      
    # WRITE OUR DATA OUT TO DATABASE OR A CSV
    
    if settings.database_enabled==True:
      # write to database
      
      data={}
      data['ticker']=ticker['ticker']
      data['price']=pr
      data['50_DAY_MA']=pr1
      data['50_DAY_DIF']=pr2
      data['200_DAY_MA']=pr3
      data['200_DAY_DIF']=pr4

      database.put_one(price_db,{'ticker':ticker['ticker']},data)
    else:
      # write to a csv file
      the_string="%s,%s,%s,%s,%s,%s\n" %(ticker['ticker'],pr,pr1,pr2,pr3,pr4)
      with open(filename, 'a') as f:
        f.write(the_string)

 
if __name__ == "__main__":
  main()