import urllib.parse
import scrapy

from tutorial.items import PdfRoomItem

class PdfRoomSpider(scrapy.Spider):
    name = "pdf-room"
    allowed_domains = ["pdfroom.com"]
    start_urls = [
        "https://www.pdfroom.com/"
    ]

    def parse(self, response, **kwargs):
        for sel in response.xpath('//div[@class="mt-0 ml-4 p-2"]'):
            href_sel, title_sel = sel.xpath('a/@href').extract(), sel.xpath('a/div/text()').extract()
            if href_sel and title_sel:
                item = PdfRoomItem()
                item['title'] = urllib.parse.urljoin(response.url, href_sel[0])
                item['link'] = title_sel[0].strip()
                yield item
