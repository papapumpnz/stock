#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import scrapy
import datetime
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from scrapy.utils.markup import remove_tags
from items import NewsItem
import database
import urlparse
from datetime import datetime,date,timedelta


# for development only
import csv

# setup logger
#logging.config.fileConfig('log.conf')
logger = logging.getLogger(__name__)
configure_logging(install_root_handler=False)


# --------------------I N F O -----------------------------------------
 
   # This should return an item to the pipeline that has:
   #   item['url']                string - url of the actual news item html page
   #   item['se']                string - tickers stock exhchange
   #   item['ticker']            string - ticker of the stock
   #   item['date']                string - date the news item was posted in YYYY-MM-DD
   #   item['text']                string - actual news item text
   

#---------------------C L A S S E S ----------------------------------- 

class mySpider(scrapy.Spider):
    name='nzx'
    base_url='https://www.nzx.com/companies/'
    url_append='/announcements'
    settings = get_project_settings()
    output_to_file=False
    
    def start_requests(self):
        
        # connect to tickers db
        connection=database.connect(self.settings['TICKERS_MONGODB_SERVER'],self.settings['TICKERS_MONGODB_PORT'],self.settings['TICKERS_MONGODB_DB'],self.settings['TICKERS_MONGODB_COLLECTION'],self.settings['TICKERS_MONGODB_UNIQ_KEY'])

        self.logger.info("Starting %s spider crawl" % self.name)
        # pull ticker name from database
        # using csv file for time being * development only
        
        if self.settings['DATABASE_ENABLED']:
            logger.info("Reading tickers from database")
            
            #development only
            #request=scrapy.Request('https://www.nzx.com/companies/CDI/announcements', callback=self.parse)
            #request.meta['item']={'ticker':'CDI','se':'NZ'}
            #yield request
            
            db_objects=database.get_all(connection)
            
            for record in db_objects:
                url=urlparse.urljoin(self.base_url,str(record['ticker'])+self.url_append)
                self.logger.debug("Processing URL %s" % url)
                request=scrapy.Request(url, callback=self.parse)
                request.meta['item']=record
                yield request
        else:
            logger.info("Reading tickers from CSV file")
            f=open('items.csv', 'rb')
            reader=csv.DictReader(f)
            for row in reader:
                url=urlparse.urljoin(base_url,str(row['ticker'])+self.url_append)
                logger.debug("Processing URL %s" % url)
                self.logger.info("Scraping URL %s" %url)
                
                request=scrapy.Request(url, callback=self.parse)
                request.meta['item']=row
                yield request

    def parse(self, response):
        # connect to trainer DB
        connection=database.connect(self.settings['TRAINER_MONGODB_SERVER'],self.settings['TRAINER_MONGODB_PORT'],self.settings['TRAINER_MONGODB_DB'],self.settings['TRAINER_MONGODB_COLLECTION'],self.settings['TRAINER_MONGODB_UNIQ_KEY'])
        # get item passed in
        ticker=response.meta['item']
        
        # dump our entire page * development only
        page = response.url.split("/")[3]
        if self.output_to_file:
            filename = 'nzx-%s-%s.html' % (page,ticker['ticker'])
            with open(filename, 'wb') as f:
                f.write(response.body)
            self.logger.info('Saved file %s' % filename)
        
        # get news item urls
        
        news_items=response.xpath('//*[@id="body"]/tr')
        #print(news_items)
        #print(ticker)
        for news in news_items:
            # get url and date of news item
            news_item = NewsItem()
            news_item['ticker']=ticker['ticker']
            news_item['se']=ticker['se']
            news_item['url']=urlparse.urljoin(self.base_url,news.xpath('td[1]/a/@href').extract()[0].strip())
            
            self.logger.info("Processing URL %s" % news_item['url'])
            
            # check if we already have this key in our database, if so skip we dont want to process it again
            existing=database.get_one(connection,{self.settings['TRAINER_MONGODB_UNIQ_KEY']:news_item[self.settings['TRAINER_MONGODB_UNIQ_KEY']]})
            print('cats')
            if not existing:
                date=news.xpath('td[2]/text()').extract()[0].strip()
                from_date=datetime.strptime(date,'%d %b %Y,  %I:%M%p').date()
                news_item['date']=str(from_date)
                                
                # get the actual news text
                logger.info('Retrieving news item for url %s' % news_item['url'])
                request=scrapy.Request(news_item['url'], callback=self.parse_news_text)
                request.meta['item']=news_item
                yield request
                
            else:
                self.logger.debug("URL %s already been processed" % news_item['url'])
                pass
            
    def parse_news_text(self,response):
        # gets actual news item text
        item=response.meta['item']
        
        # get detail section
        detail=response.xpath('/html/body/section/div[2]/div/div/div[2]').extract()[0]
        
        # TO DO
        # is there any attachements?
        attachments=response.xpath('//*[@id="attachments"]/ul/li')
        logger.debug('%s attachements on page %s' % (len(attachments),response.url))
        for attachement in attachments:
            iclass=attachement.xpath('@class').extract()
            #print(iclass)
        
        # strip html tags
        item['text']=remove_tags(detail)
        
        yield item
    
    def parse_attachment(self,response):
        # get any text from attachements on the page
        pass