# settings file

# database overrride. If false, write to CSV file
database_enabled=True

# ticker database
ticker_db_server = 'localhost'
ticker_db_port = 27017
ticker_db_database = 'stock'
ticker_db_collection = 'tickers'
ticker_db_uniq_key = 'ticker'

# prices database
price_db_server = 'localhost'
price_db_port = 27017
price_db_database = 'stock'
price_db_collection = 'price'
price_db_uniq_key = 'ticker'

# proxy
proxy_enabled=False
proxy_username='marshs'
proxy_password='enc(q29zbw8ymdkw)'
proxy_server='nzproxy.lb.service.anz:80'