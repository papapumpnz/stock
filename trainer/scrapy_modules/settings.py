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

# Database General stuff
DATABASE_ENABLED=True

# Database Tickers
TICKERS_MONGODB_SERVER = 'localhost'
TICKERS_MONGODB_PORT = 27017
TICKERS_MONGODB_DB = 'stock'
TICKERS_MONGODB_COLLECTION = 'tickers'
TICKERS_MONGODB_UNIQ_KEY = 'ticker'

# Database trainer
TRAINER_MONGODB_SERVER = 'localhost'
TRAINER_MONGODB_PORT = 27017
TRAINER_MONGODB_DB = 'stock'
TRAINER_MONGODB_COLLECTION = 'trainer'
TRAINER_MONGODB_UNIQ_KEY = 'url'

# Price History days - amount of days to get historical price data for a news item
PRICE_HISTORY_DAYS=3

# Classification rules
CLASSIFICATON_POST_NEWS_TRADE_VOL_GRT=0			# dont classify news items that have trade volumes less than or equal to