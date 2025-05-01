#!/usr//bin/env/python3
# -*- coding:utf-8 -*-
__author__ = 'Hiram Zhang'

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from app.utils import getPinyin


def data_grabber_3u(dept, arv, flight_date, proxy=None,
                    executable_path=r'D:\chromedriver_win32\chromedriver.exe',
                    headless=True):
    input_dep_city = dept
    input_arv_city = arv
    input_flight_date = flight_date
    options = Options()
    options.headless = headless
    if proxy is not None:
        options.add_argument('--proxy-server=%s' % proxy)
    driver = webdriver.Chrome(chrome_options=options, executable_path=executable_path)

    driver.get('http://www.sichuanair.com//')
    wait = WebDriverWait(driver, 20)
    dep_input = wait.until(
        EC.presence_of_element_located((By.ID, 'Search-OriginDestinationInformation-Origin-location_input_location')))
    arv_input = wait.until(
        EC.presence_of_element_located(
            (By.ID, 'Search-OriginDestinationInformation-Destination-location_input_location')))
    date_picker = wait.until(
        EC.presence_of_element_located((By.NAME, 'Search/DateInformation/departDate_display')))
    cookie_icon = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.cookie-policy .cookie-yes')))
    cookie_icon.click()

    dep_input.click()
    city_index = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.select_results .hotCity .title .col')))
    alphabet = getPinyin(input_dep_city)
    if 'f' >= alphabet >= 'a':
        target_city_index = city_index[1]
    elif 'g' >= alphabet >= 'j':
        target_city_index = city_index[2]
    elif 'n' >= alphabet >= 'k':
        target_city_index = city_index[3]
    elif 'w' >= alphabet >= 'p':
        target_city_index = city_index[4]
    elif 'z' >= alphabet >= 'x':
        target_city_index = city_index[5]

    target_city_index.click()
    city_list = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.select_results .hotCity .city li')))
    for i in city_list:
        if i.text == input_dep_city:
            i.click()
            break

    arv_input.click()
    city_index = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.select_results .hotCity .title .col')))
    alphabet = getPinyin(input_arv_city)
    if 'f' >= alphabet >= 'a':
        target_city_index = city_index[1]
    elif 'g' >= alphabet >= 'j':
        target_city_index = city_index[2]
    elif 'n' >= alphabet >= 'k':
        target_city_index = city_index[3]
    elif 'w' >= alphabet >= 'p':
        target_city_index = city_index[4]
    elif 'z' >= alphabet >= 'x':
        target_city_index = city_index[5]

    target_city_index.click()
    city_list = wait.until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.select_results .hotCity .city li')))
    for i in city_list:
        if i.text == input_arv_city:
            i.click()
            break

    date_picker.send_keys(Keys.CONTROL, 'a')
    date_picker.send_keys(Keys.BACKSPACE)
    date_picker.send_keys(input_flight_date)
    btn = driver.find_element(By.CSS_SELECTOR, '.ui-state-active, .ui-widget-content .ui-state-active')
    btn.click()

    driver.execute_script('submitForm();')
    tb_head = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '.tbh-section')))

    result = []
    if '组合价' in tb_head.text:
        return result
    if '您所查询的航线没有找到适用价格' in driver.page_source:
        return result

    soup = BeautifulSoup(driver.page_source, 'html5lib')
    l_flt = soup.select('.brand-tb .tbd-section')

    for flt in l_flt:
        flt_no = flt.select_one('.route-info .flight-code').get_text().strip()
        share_info = flt.select_one('.route-info .air-code .hover-con .hover-con-wrap')
        is_shared = 0
        share_company = None
        share_flight_no = None
        if share_info:
            is_shared = 1
            share_info_clear = share_info.get_text().split('：')[1].split('，')[0]
            share_company = share_info_clear[:4].strip()
            share_flight_no = share_info_clear[4:].strip()
        airplane_type = flt.select_one('.route-info .plane-type .plane-trigger').get_text().strip()
        dep_time = flt.select_one('.route-start-end .route-start .route-time').get_text().strip()
        dep_airport = flt.select_one('.route-start-end .route-start .route-place').get_text(). \
            strip().replace('\t', '').replace('\n', '').replace(' ', '').replace('\xa0', '')
        dep_date = flt.select_one('.route-start-end .route-start .route-date').get_text().strip()
        flt_time = flt.select_one('.route-start-end .route-to .period-time').get_text(). \
            strip().replace('\t', '').replace('\n', '').replace(' ', '')
        is_direct = 1
        transfer_city = None
        direct_tag = flt.select_one('.route-start-end .route-to .i-direct')
        if direct_tag is None:
            is_direct = 0
            direct_info = flt.select_one('.route-start-end .route-to .stay-place span')
            if direct_info:
                transfer_city = direct_info.get_text().strip()
            else:
                transfer_city = '未知'
        arv_time = flt.select_one('.route-start-end .route-end .route-time').get_text().strip()
        arv_airport = flt.select_one('.route-start-end .route-end .route-place').get_text(). \
            strip().replace('\t', '').replace('\n', '').replace(' ', '').replace('\xa0', '')
        arv_date = flt.select_one('.route-start-end .route-end .route-date').get_text().strip()
        price_list = flt.select('.tb-td.price-td')
        price_index = ['公务舱', '标准经济舱', '优选经济舱', '优惠经济舱', '超值经济舱']
        price_l = []
        for i, price in enumerate(price_list):
            if 'no-price-td' in price.attrs['class']:
                continue
            price_type1 = price_index[i]
            price_type2 = price_index[i]
            price_value = price.select_one('.ticket-wrap .ticket-price').get_text().strip()[1:]
            price_l.append((price_type1, price_type2, price_value))
            result.append(
                dict(dep_city=input_dep_city, arv_city=input_arv_city, is_direct=is_direct, transfer_city=transfer_city,
                     flt_no=flt_no, airplane_type=airplane_type,
                     is_shared=is_shared, share_company=share_company, share_flight_no=share_flight_no,
                     dep_date=dep_date, dep_time=dep_time, arv_date=arv_date, arv_time=arv_time, flt_time=flt_time,
                     dep_airport=dep_airport, arv_airport=arv_airport,
                     price_type1=price_type1, price_type2=price_type2, price_value=price_value))

    return result


if __name__ == '__main__':
    rs = data_grabber_3u('北京', '丽江', '2019-01-15', headless=False)
    for i in rs:
        print(i)
