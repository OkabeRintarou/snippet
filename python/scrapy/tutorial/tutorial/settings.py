BOT_NAME = 'pdf-room'

SPIDER_MODULES = ['tutorial.spiders']
NEWSPIDER_MODULE = 'tutorial.spiders'

ROBOTSTXT_OBEY = True

ITEM_PIPELINES = {
    'tutorial.pipelines.PdfRoomPipeline': 300,
}
