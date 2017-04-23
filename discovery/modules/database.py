#!/usr/bin/env python
# -*- coding: utf-8 -*-

# database related functions

import logging
logger = logging.getLogger(__name__)
from scrapy.utils.project import get_project_settings
settings = get_project_settings()

try:
  import pymongo
except ImportError as e:
  logger.error("Error importing pymongo library. Is it installed?")

# connect to the database
connection = pymongo.MongoClient(settings['MONGODB_SERVER'], settings['MONGODB_PORT'])
db = connection[settings['MONGODB_DB']]
collection = db[settings['MONGODB_COLLECTION']]

def read_database(filter=None):
  # read in rows from the database
  
  if filter is not None:
    return collection.find({filter})
  else:
    return collection.find()