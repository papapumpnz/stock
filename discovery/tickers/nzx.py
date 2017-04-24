#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import scrapy
from scrapy.utils.log import configure_logging
from scrapy.selector import Selector
from items import TickerItem
import urlparse

# setup logger
#logging.config.fileConfig('log.conf')
logger = logging.getLogger(__name__)
configure_logging(install_root_handler=False)

#---------------------C L A S S E S ----------------------------------- 

class mySpider(scrapy.Spider):
    name='nzx'
    se_name='NZ'
    base_url='https://www.nzx.com'
    base_url_info='https://www.nzx.com/companies/'
    
    def start_requests(self):
        urls = [
            'https://www.nzx.com/markets/NZSX/securities',
        ]
        for url in urls:
            request=scrapy.Request(url, callback=self.parse)
            yield request

    def parse(self, response):
        self.logger.info("Starting %s spider crawl" % self.name)
        
        # dump our entire page * development only
        page = response.url.split("/")[-2]
        filename = 'nzx-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.logger.info('Saved file %s' % filename)
        
        # process it
        sel = Selector(response)
        tickers=sel.xpath('//*[@id="instruments"]/table/tbody/tr')

        self.logger.info("Found %s tickers" % len(tickers))
        for ticker in tickers:
          #print(ticker)
          item=TickerItem()
          item['ticker']=ticker.xpath('td[1]/a/text()').extract()[0].strip()
          item['se_url']=ticker.xpath('td[1]/a/@href').extract()[0].strip()
          item['se']=self.se_name
          item['stock_name']=ticker.xpath('td[2]/@alt').extract()[0].strip()

          # generate company info url
          url=urlparse.urljoin(self.base_url_info,str(item['ticker']))

          # fetch it
          request=scrapy.Request(url, callback=self.parse_info)
          request.meta['item']=item
          yield request
          

    
    def parse_info(self,response):
        item=response.meta['item']

        item['company_name']=response.xpath('/html/body/section/div[2]/div/div/div[2]/header/h2/text()').extract()[0].strip()
        item['company_url']=response.xpath('/html/body/section/div[2]/div/div/div[2]/div[1]/dl/dd[5]/a/@href').extract()[0].strip()
        
        yield item
