import scrapy


class PdfRoomItem(scrapy.Item):
    title = scrapy.Field()
    link = scrapy.Field()
