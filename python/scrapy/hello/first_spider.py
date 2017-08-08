from scrapy.spiders import Spider

class FirstSpider(Spider):
		name = 'helloworld'
		start_urls = ['https://dirtysalt.github.io']

		def parse(self,response):
				contents = response.xpath('//a/text()').extract()
				for content in contents:
					print(content)
