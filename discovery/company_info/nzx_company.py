#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import scrapy
import datetime
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from items import TickerItem
import database

# for development only
import csv

# setup logger
#logging.config.fileConfig('log.conf')
logger = logging.getLogger(__name__)
configure_logging(install_root_handler=False)


#---------------------C L A S S E S ----------------------------------- 

class mySpider(scrapy.Spider):
    name='nzx_company'
    base_url='https://www.nzx.com/companies/'
    settings = get_project_settings()
    output_to_file=False
    
    def start_requests(self):
        
        self.logger.info("Starting %s spider crawl" % self.name)
        # pull ticker name from database
        # using csv file for time being * development only
        
        if self.settings['DATABASE_ENABLED']:
          logger.info("Reading tickers from database")
          db_objects=database.read_database()
          for record in db_objects:
            url='%s%s' % (self.base_url,str(record['ticker']))
            #logger.debug("Processing URL %s" % url)
            self.logger.info("Scraping URL %s" %url)
            yield scrapy.Request(url=url, callback=self.parse)          
        else:
          logger.info("Reading tickers from CSV file")
          f=open('items.csv', 'rb')
          reader=csv.DictReader(f)
          for row in reader:
            url='%s%s' % (self.base_url,row['ticker'])
            #logger.debug("Processing URL %s" % url)
            self.logger.info("Scraping URL %s" %url)
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        
        # dump our entire page * development only
        #print(response.url.split("/"))
        page = response.url.split("/")[3]
        ticker = response.url.split("/")[4]
        if self.output_to_file:
          filename = 'nzx-%s-%s.html' % (page,ticker)
          with open(filename, 'wb') as f:
              f.write(response.body)
          self.logger.info('Saved file %s' % filename)
        
        # process it
        item = TickerItem()
        item['ticker']=ticker
        item['company_name']=response.xpath('/html/body/section/div[2]/div/div/div[2]/header/h2/text()').extract()[0].strip()
        item['company_url']=response.xpath('/html/body/section/div[2]/div/div/div[2]/div[1]/dl/dd[5]/a/@href').extract()[0].strip()
        #print(item)
        return item