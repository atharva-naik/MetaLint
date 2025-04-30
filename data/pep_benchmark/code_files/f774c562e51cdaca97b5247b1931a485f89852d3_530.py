from douyin.structures import Music
from douyin.handlers import Handler
from douyin.downloaders import Downloader


class MusicDownloader(Downloader):
    
    async def process_item(self, obj):
        """使用异步开始下载音乐
        process item
        :param obj: single obj
        :return:
        """
        if isinstance(obj, Music):
            print('Processing', obj, '...')
            for handler in self.handlers:
                if isinstance(handler, Handler):
                    await handler.process(obj)
