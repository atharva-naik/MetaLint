from flask import Flask
import flask_profiler
import http.client
from itertools import cycle


app = Flask(__name__)
app.config["DEBUG"] = True
app.config["flask_profiler"] = {
    "enabled": app.config["DEBUG"],
    "storage": {
        "engine": "sqlite",
    },
    "basicAuth":{
        "enabled": True,
        "username": "admin",
        "password": "admin"
    }
}

class RoundRobin():
    next_elem = {}
    ports = [5001, 5002, 5003]
    ports_cycle = cycle(ports)

    def get_next_port(self):
        self.next_elem = next(i.ports_cycle)
        return self.next_elem


i = RoundRobin()


@app.route('/calculate_pi/<id>')
def balance_request(id):
    conn = http.client.HTTPConnection("www.localhost:" + str(i.get_next_port()))
    conn.request("GET", "/calculate_pi/" + str(id))
    r1 = conn.getresponse()
    data = r1.read()
    return data


flask_profiler.init_app(app)
