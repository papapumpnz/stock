# Define your item pipelines here
#

from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
import logging
from datetime import datetime

# setup logger
#logging.config.fileConfig('log.conf')
logger = logging.getLogger(__name__)
configure_logging(install_root_handler=False)

# Copyright 2011 Julien Duponchelle <julien@duponchelle.info>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class MongoDBPipeline(object):

    settings = get_project_settings()
    
    def open_spider(self,spider):
      if self.settings['DATABASE_ENABLED']:
        try:
          import pymongo
        except ImportError as e:
          logger.error("Error importing pymongo library. Is it installed?")

        logger.info("Establishing connection to mongoDB %s:%s" % (self.settings['MONGODB_SERVER'], self.settings['MONGODB_PORT']))
        self.connection = pymongo.MongoClient(self.settings['MONGODB_SERVER'], self.settings['MONGODB_PORT'])
        self.db = self.connection[self.settings['MONGODB_DB']]
        self.collection = self.db[self.settings['MONGODB_COLLECTION']]
        if self.__get_uniq_key() is not None:
            try:
              logger.debug("Creating database index '%s'" %self.__get_uniq_key())
              self.collection.create_index(self.__get_uniq_key(), unique=True)
            except pymongo.errors.ServerSelectionTimeoutError as e:
              logger.error("Unable to connect to mongoDB server %s. Error: %s" %(self.settings['MONGODB_SERVER'],e))
            except Exception as e:
              logger.error("An error occured. Error: %s" %(e))
    
    def close_spider(self,spider):
      self.connection.close()
      logger.info("Closed connection to mongoDB %s:%s" % (self.settings['MONGODB_SERVER'], self.settings['MONGODB_PORT']))
      
    def process_item(self, item, spider):
        #print(item)
        if self.settings['DATABASE_ENABLED']:
          if self.__get_uniq_key() is None:
              try:
                logger.debug("Attempting insert new record for ticker %s" %item[self.__get_uniq_key()])
                result=self.collection.insert_one(dict(item))
                logger.debug("Inserted %i record" % result.modified_count)
              except pymongo.errors.ServerSelectionTimeoutError as e:
                logger.error("Unable to connect to mongoDB server %s. Error: %s" %(self.settings['MONGODB_SERVER'],e))
              except Exception as e:
                logger.error("An error occured. Error: %s" %(e))
          else:
              #print("Item: %s" % item)
              existing=self.collection.find_one({self.__get_uniq_key():item[self.__get_uniq_key()]})
              #print("Existing: %s" % existing)
              if existing is not None:
                logger.debug("Attempting update existing record for ticker %s" % existing[self.__get_uniq_key()])
                # merge item and existing and save
                #new_item=self.__merge_dicts(existing,item)
                #print("New item : %s" % new_item)
                
                #item['updated']=str(datetime.now())
                new_item=dict(item)
                new_item['updated']=str(datetime.now())
                try:
                  #result=self.collection.update_one({"ticker":"ABA"},{"$set":{"company_url":"test"}})
                  result=self.collection.update_one({self.__get_uniq_key():existing[self.__get_uniq_key()]},{'$set':dict(new_item)})
                except pymongo.errors.ServerSelectionTimeoutError as e:
                  logger.error("Unable to connect to mongoDB server %s. Error: %s" %(self.settings['MONGODB_SERVER'],e))
                except Exception as e:
                  logger.error("An error occured. Error: %s" %(e))
              else:
                logger.debug("Attempting insert new record for ticker %s" %item[self.__get_uniq_key()])
                new_item=dict(item)
                new_item['inserted']=str(datetime.now())
                try:
                  result=self.collection.insert_one(dict(new_item))
                except pymongo.errors.ServerSelectionTimeoutError as e:
                  logger.error("Unable to connect to mongoDB server %s. Error: %s" %(self.settings['MONGODB_SERVER'],e))
                except Exception as e:
                  logger.error("An error occured. Error: %s" %(e))
                  
        return item

    def __get_uniq_key(self):
        if not self.settings['MONGODB_UNIQ_KEY'] or self.settings['MONGODB_UNIQ_KEY'] == "":
            return None
        return self.settings['MONGODB_UNIQ_KEY']
          