import itchat
import pymysql
import re



def sqlQuery(query):
    conn = pymysql.connect(host='192.168.31.168', port=3307, user='test', password='123456', db='test',
                           use_unicode=True)
    cursor = conn.cursor()
    cursor.execute(query)
    values = cursor.fetchall()
    conn.close()
    return values
# reply msg depends on what i got

# record users
users = []


@itchat.msg_register(itchat.content.TEXT)
def text_reply(msg):
    # Need admin function first.Use the key to add to users, then could use the function
    if "刘主任辛苦了" in msg.text:
        users.append(msg['FromUserName'])
        itchat.send_msg("小同志你说的很对，我看你很有前途嘛,不会用可以说功能嘛，用完了可以说再见，要不然会一直自动回复你", msg['FromUserName'])
        return
    if msg['FromUserName'] not in users:
        return
    else:
        if "查单" in msg.text:
            num = re.sub("查单", "", msg.text)
            if num == "":
                for row in sqlQuery('select * from orderStatus order by id desc limit 10 '):
                    itchat.send_msg('利达单号'+ row[1] + ' ,国内快递号'+row[2], msg['FromUserName'])
            else:
                for row in sqlQuery('select * from orderStatus order by id desc limit'+num):
                    itchat.send_msg('利达单号'+ row[1] + ' ,国内快递号'+row[2], msg['FromUserName'])
        elif "查买" in msg.text:
            # itemName = re.sub("查买", "",msg.text)
            for row in sqlQuery('select * from Products order by P_id desc limit 10'):
                itchat.send_msg('最近需要购买'+ row[1]+' ,共需要' + str(row[2]), msg['FromUserName'])
        elif "买" in msg.text:
            print(msg.text)
            reString = re.sub("买", "", msg.text)
            itemQua = re.findall(u"([^\u4e00-\u9fa5])", reString)
            itemName = re.findall(u"([\u4e00-\u9fa5])", reString)
            itemQua = ''.join(itemQua)
            itemName = ''.join(itemName)
            sql = """insert into Products (Pname, Pquality) values (  """ + "\""+itemName+"\"" + ", " + itemQua + ")"
            conn = pymysql.connect(host='192.168.31.168', port=3307, user='test', password='123456', db='test',
                                   use_unicode=True)
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            conn.close()
            itchat.send_msg(itemName + itemQua, msg['FromUserName'])
        elif msg.text == "功能":
            itchat.send_msg('回复查单+数字显示最近单号，默认显示10，例:查单 15显示最近15个订单', msg['FromUserName'])
            itchat.send_msg('回复查买+数字显示最近下单买的东西，默认显示是，例：查买 15显示最近15个需要购买的状态', msg['FromUserName'])
            itchat.send_msg('回复买+物品+数量,下单购买物品，例：买 小熊糖 10', msg['FromUserName'])
            itchat.send_msg('回复再见退出神奇的下单模式', msg['FromUserName'])
            # itchat.send_msg(msg['FromUserName'], msg['FromUserName'])
        elif msg.text == "再见":
            users.remove(msg['FromUserName'])
            itchat.send_msg('小同志忙完了？再会再会~记得我们的接头暗号：刘主任辛苦了，要不然没人搭理你哦', msg['FromUserName'])
            return
        else:
            itchat.send_msg('没有找到指定命令，请发送 功能 查看具体命令', msg['FromUserName'])


if __name__ == "__main__":
    itchat.auto_login()
    itchat.run()