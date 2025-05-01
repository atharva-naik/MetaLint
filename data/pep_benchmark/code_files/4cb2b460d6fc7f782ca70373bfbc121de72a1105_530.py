import asyncio
import aiohttp
from lxml import etree
import pprint
import re
import json

start_url = 'https://new.qq.com/ch/finance/'
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
}
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'
headers.update({'User-Agent': user_agent})

base_params = {
    'chlid': 'news_rss',
    'refer': 'mobilewwwqqcom',
    'otype': 'jsonp',
    'ext_data': 'all',
    'srcfrom': 'newsapp',
    'callback': 'getNewsContentOnlyOutput',
}

pp = pprint.PrettyPrinter(indent=4)


async def start_request(session):
    async with session.get(start_url, headers=headers) as response:
        status_code = response.status
        if 200 == status_code:
            print('200 OK!')
            ret_list = []
            selector = etree.HTML(await response.text())
            a_tags = selector.xpath('//div[@class="main"]/div[@class="head"]/div[@class="Q-tpList"]/div/div/em/a')
            for a in a_tags:
                link = a.attrib['href']
                news_id = re.findall(r'/([^/]+)$', link)
                if news_id:
                    ret_list.append({
                        'title': a.text,
                        'link': news_id[0]
                    })
            return {
                'success': 1,
                'result': ret_list
            }
        return {
            'success': 0,
            'error_code': status_code
        }


async def get_list(url, session):
    link = 'https://openapi.inews.qq.com/getQQNewsNormalContent'
    title = url['title']
    params = base_params.copy()
    params['id'] = url['link']
    async with session.get(link, headers=headers, params=params) as response:
        status_code = response.status
        try:
            assert 200 == status_code
            print('200 OK!')
            reg = re.compile(r'getNewsContentOnlyOutput\(([\w\W]+)\)$')
            json_str = re.findall(reg, await response.read())[0] if re.findall(reg, await response.text(encoding='gbk')) else None
            json_obj = json.loads(json_str) if json_str else None
            text = ''
            if json_obj and json_obj['content']:
                for content in json_obj['content']:
                    if content['type'] == 1:
                        text += content['value']
            else:
                text = 'no content.'
            pp.pprint({
                'title': title,
                'content': text,
            })
        except Exception as err:
            print('title "%s" errors: %s' % (title, err))


async def run(loop):
    async with aiohttp.ClientSession(loop=loop) as sess:
        res = await start_request(sess)
        if 1 == res['success']:
            urls = res['result']
            for url in urls:
                await get_list(url, sess)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    loop.close()

