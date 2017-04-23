#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from scrapy.crawler import CrawlerRunner,Crawler
from scrapy.conf import settings
from scrapy.utils.log import configure_logging
from twisted.internet import reactor, defer
from scrapy import signals
import importlib
import base64
import re

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
    d = self.process.join()
    d.addBoth(lambda _: reactor.stop()) # call this when all spiders are closed
    reactor.run() # the script will block here until the crawling is finished
 
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
  process = CrawlerRunner(settings)
  
  # create a spider object
  spider=Spiders(logger,process,settings)
  
  ###
  #   TICKERS CRAWL
  ###
  
  # load in tickers spiders
  spider.findSpiders('tickers')

  ###
  #   COMP INFO CRAWL
  ###  

  # load in company info spiders
  spider.findSpiders('company_info') 
  
  # list spiders
  logger.info("Spiders loaded are:")
  spider.listSpiders()

  # run all spiders
  spider.startAllSpiders() # the script will block here until the last crawl call is finished
  
  
  # finish
  logger.info("All spiders finished")
 
if __name__ == "__main__":
  main()