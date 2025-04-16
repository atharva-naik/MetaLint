#!/usr/bin/python3
# -*-coding: utf-8-*-


import requests
import random
import string


def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(n))


def create_users(number=10):
    for i in range(number):
        name, password = random_string(8), random_string(15)
        r = requests.post("http://127.0.0.1:8888/api/v1/users", data={'name': name, 'password': password})
        print(r.status_code)


def create_project(name, user_id):
    r = requests.post("http://127.0.0.1:8888/api/v1/project", data={'name': name, 'user_id': user_id})
    print(r.status_code)


def create_rss_links(project_id):
    links = [
        ("nextinpact", "http://www.pcinpact.com/rss/news.xml"),
        ("reflets", "http://reflets.info/feed/"),
        ("factornews", "http://www.factornews.com/rss.php")
    ]

    for name, link in links:
        r = requests.post("http://127.0.0.1:8888/api/v1/rss_links", data={'name': name, "link": link, "project_id": project_id})
        print(r.text)


def get_feeds(feed_id):
    url = "http://127.0.0.1:8888/api/v1/rss_links/{}/feeds".format(feed_id)
    r = requests.get(url)
    feeds = r.json()
    for feed in feeds['feeds']:
        print(feed.get('summary'))

if __name__ == "__main__":
    # create_users()
    # create_project("project1", 1)
    # create_rss_links(1)
    get_feeds(2)

