import tornado.ioloop
import tornado.web
import time,hashlib
from controllers.account import LoginHandler
from controllers.home import HomeHandler

container = {}
# container = { '随机字符串或者cookie':{'uuuu':'root','k1':'v1'}, }

class Session(object):
    def __init__(self,handler):
        self.handler = handler
        self.random_str = None              # 随机字符串，也有可能是cookie
        self.client_random_str = self.handler.get_cookie('session_id')

        if not self.client_random_str:
            """新用户，没有cookie的"""
            # 1. 生成随机字符串当作session中的key,保存在大字典container中，用户每次访问都到里面找是否有该值
            self.random_str = self.create_random_str()
            container[self.random_str] = {}                 # 保存在大字典
        else:
            if self.client_random_str in container:
                """老用户，在container大字典里面了"""
                self.random_str = self.client_random_str
                print('老用户',container)
            else:
                """非法用户，伪造的cookie"""
                self.random_str = self.create_random_str()
                container[self.random_str] = {}

        # 2. 生成cookie，必须调用LoginHandler才能使用set_cookie()
        timeOut = time.time()
        self.handler.set_cookie('session_id',self.random_str,expires=timeOut+1800)

        # 3. 写入缓存或数据库  ==> 后面用户自己调用session['uuuu'] = 'root'

    def create_random_str(self):
        now = str(time.time())
        m = hashlib.md5()
        m.update(bytes(now,encoding='utf-8'))
        return m.hexdigest()

    def __setitem__(self, key, value):
        # print(key,value)                 # key 就是用户自己设置session['uuuu']='root'中的uuuu，value就是root
        container[self.random_str][key] = value
        # print('setitem',container)

    def __getitem__(self, item):
        # print(item)                       # uuuu
        # print('getitem',container)
        return container[self.random_str].get(item)

    def __delitem__(self, key):
        pass
    def open(self):
        pass
    def cloes(self):
        pass

class Foo(object):
    def initialize(self):
        # print(self)         # <__main__.LoginHandler object at 0x00000000038702E8>
        self.session = Session(self)
        super(Foo, self).initialize()       # 执行RequestHandler中的initialize

class HomeHandler(Foo,tornado.web.RequestHandler):
    def get(self):
        print('session',self.session)
        user = self.session['uuuu']         # 调用Session类中的__getitem__方法, 获取value
        print('user',user)
        if not user:
            self.write('不是合法登录')
        else:
            self.write(user)

class LoginHandler(Foo,tornado.web.RequestHandler):
    def get(self):

        # self.session['uuuu']                  # 调用Session类中的__getitem__方法, 获取value
        # del self.session['uuuu']              # 调用Session类中的__delitem__方法, 删除
        self.session['uuuu'] = "root"           # 调用Session类中的__setitem__方法，在session里面设置了uuuu
        self.write("Hello, world")
        print(container)
        self.redirect('/home')



application = tornado.web.Application([
    # (r"/index", MainHandler),
    (r"/login", LoginHandler),
    (r"/home", HomeHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()