#!/usr/bin/env python
# -*- coding: utf-8 -*-

# database related functions

import logging
logger = logging.getLogger(__name__)
from scrapy.utils.project import get_project_settings
settings = get_project_settings()
from datetime import datetime

try:
  import pymongo
except ImportError as e:
  logger.error("Error importing pymongo library. Is it installed?")

def connect(server,port,database,collection,index):
  # connect to the database
  connection = pymongo.MongoClient(server, port)
  db = connection[database]

  collection=db[collection]
  
  # setup index
  if index:
	collection.create_index(index, unique=True)

  return collection

def get_one(collection,filter=None):
  # read in rows from the database
  
  #print('Filter:%s' % filter)
  
  if filter is not None:
    return collection.find_one(dict(filter))
  else:
    return collection.find()

def get_all(collection,filter=None):
  # read in rows from the database
  
  #print('Filter:%s' % filter)
  
  if filter is not None:
    return collection.find(dict(filter))
  else:
    return collection.find()


def put_one(collection,index,data):
    # write in rows from the database
    # index : index, ie {'ticker':'BFA'}
    # data : dict of data to write, ie {'URL':'http://blah.com','Date':'2017-04-01:12:00:00'}

    result=None

    #print('Index:%s' % index)
  
    if index:
        # first check if index already exists, if so update instead of insert
        if get_one(collection,index):
            # update exist record
            data['updated']=str(datetime.now())
            result=collection.update_one(dict(index),{'$set':dict(data)})
        else:
            # insert new record
            data['inserted']=str(datetime.now())
            result=collection.insert_one(dict(data))
    else:
        # insert new record
        data['inserted']=str(datetime.now())
        result=collection.insert_one(dict(data))
    
	return result