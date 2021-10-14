import json

import pymysql


class PdfRoomPipeline:
    def __init__(self):
        self.conn = pymysql.connect(host='localhost', port=3306, user='syl', password='123321', database='hello')
        self.cur = self.conn.cursor()

    def process_item(self, item, spider):
        sql_word = "insert into book(title, link) values(\"{0}\", \"{1}\");".format(item["title"], item["link"])
        print(sql_word)
        self.cur.execute(sql_word)
        self.conn.commit()
        return item
