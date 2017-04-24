from scrapy.item import Item, Field

class NewsItem(Item):
	# Here are the fields that will be crawled and stored
	ticker = Field() # ticker
	se = Field() # stock exchange of ticker
	url = Field() # url of news item
	text = Field() # text of news item
	date = Field() # date of news item
	classification = Field() # neg or pos classification of news item
	price_history = Field() # stock price diff following 3 days of news item, ie -5% or +5%
	