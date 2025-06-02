"""
Actions For Skills Of Alexa
"""
import datetime, psycopg2


class Action():
    """
    Action class
    """

    def __init__(self):
        # Database Connection
        self.conn = psycopg2.connect(database="AlexaDB", user="postgres",\
                        password="postgres", host="127.0.0.1", port="5432")
        self.cur = self.conn.cursor()
        self.balance = 0

    def greetings(self):
        """
        Greeting action
        """
        hour= datetime.datetime.now().hour
        reply = ''
        if hour>=0 and hour<12:
            reply = "Hi, Good morning."
        elif hour>=12 and hour<16:
            reply = "Hi, Good Afternoon."
        elif hour>=16 and hour<24:
            reply = "Hi, Good Evening."
        else:
            reply = "Hi, nice to meet you."
        return reply

    def get_customer_balance(self, customer_id):
        """
        Action to retrieve balance of a customer
        """
        try:
            self.cur.execute("select balance from customer where customer_id='"+str(customer_id)+"'")
            result = self.cur.fetchone()
            self.balance = str(result[0])
            return self.balance
        except Exception as e:
            print("Failed due to ", e)