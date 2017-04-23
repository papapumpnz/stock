# SPIDER
BOT_NAME = 'fetch'
USER_AGENT='Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'

# PROXY DETAILS
DOWNLOADER_MIDDLEWARES={'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 1}
PROXY_ENABLED=False
PROXY_USERNAME='marshs'
PROXY_PASSWORD='ENC(Q29zbW8yMDkw)'
PROXY_SERVER='nzproxy.lb.service.anz:80'

# Enable auto throttle
AUTOTHROTTLE_ENABLED = True

# Item pipeline
ITEM_PIPELINES = {
                  'pipelines.MongoDBPipeline':1,
}

# Feed Exporter to JSON * development only
FEED_FORMAT = 'csv'
FEED_URI ='items.csv'

# Database
DATABASE_ENABLED=True
MONGODB_SERVER = 'localhost'
MONGODB_PORT = 27017
MONGODB_DB = 'stock'
MONGODB_COLLECTION = 'tickers'
MONGODB_UNIQ_KEY = 'ticker'