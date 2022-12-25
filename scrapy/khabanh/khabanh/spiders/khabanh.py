import scrapy
import os
import urllib

DIR_BASE_TRAIN = r'D:\CODE\python\bees_and_aunts\data\train\khabanh'
DIR_BASE_VAL = r'D:\CODE\python\bees_and_aunts\data\val\khabanh'


class KhaBanhSpider(scrapy.Spider):
    name = 'khabanh'
    allowed_domain = ['google.com']

    def __init__(self):
        super().__init__()
        self.check = None
        self.list_links = None

    def start_requests(self):
        first_url = '''https://www.google.com/search?q=kh%C3%A1+b%E1%BA%A3nh&tbm=isch&ved=2ahUKEwiZ2Kve6pH8AhUFEKYKHU
        z3DrsQ2-cCegQIABAA&oq=kh%C3%A1+b%E1%BA%A3nh&gs_lcp=CgNpbWcQDFAAWABgAGgAcAB4AIABAIgBAJIBAJgBAKoBC2d3cy13aXotaW1n&sclient=img&ei=6bemY9m_L4WgmAXM7rvYCw&bih=615&biw=1322'''
        self.page = 0
        self.count = 0
        yield scrapy.Request(url=first_url, callback=self.parse_link)


    def parse_link(self, response):
        base_url = 'https://www.google.com/'
        for link in response.css('.yWs4tf'):
            self.get_img(link.attrib['src'], self.count)
            self.count += 1
            print(self.count)
        self.page += 1
        if self.page < 21:
            yield scrapy.Request(url =base_url + response.css('.frGj1b')[-1].attrib['href'], callback= self.parse_link)

    def get_img(self, link, count):
        dir_base = DIR_BASE_TRAIN
        if self.page > 10:
            dir_base = DIR_BASE_VAL

        image_name = 'khabanh_{}_{}'.format(count, '.jpg')
        local_path_image = os.path.join(dir_base, image_name)
        urllib.request.urlretrieve(link, local_path_image)
