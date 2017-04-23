from scrapy.item import Item, Field

class TickerItem(Item):
    # Here are the fields that will be crawled and stored
    ticker = Field() # ticker
    stock_name = Field() # company stock name
    company_name = Field() # company name
    company_url = Field() # companies own URL
    se_url = Field() # stock exchange URL
    se = Field()  # stock exchange this stock belongs to
    date = Field() # date inserted or change.