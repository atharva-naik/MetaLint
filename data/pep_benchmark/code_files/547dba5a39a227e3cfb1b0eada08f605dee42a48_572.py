import fitz
import requests
import time
import re

# English to Korean Translation Func.
def papagoAPI(id, secret, sentence):
    request_url = "https://openapi.naver.com/v1/papago/n2mt"
    headers ={"X-Naver-Client-Id": id, "X-Naver-Client-Secret": secret}
    params = {"source": "en", "target": "ko", "text": sentence}
    response =requests.post(request_url, headers=headers, data=params)
    result = response.json()

    return result["message"]["result"]["translatedText"]
'''
response.json 파일 포맷 형식
{'message': {'@type': 'response', '@service': 'naverservice.nmt.proxy', '@version': '1.0.0', 'result': {'srcLangType': 'en', 'tarLangType': 'ko', 'translatedText': '어떻게 지내니?', 'engineType': 'N2MT', 'pivot': None}}}

'''

# English PDF file to 한국어 Text파일 저장 및 읽기
def pdfToText(inputFile):
    doc = fitz.open(inputFile)
    print("문서 페이지 수 : ", len(doc))

    # pdf -> Text save.
    ext_text = ""
    temp = ""
    for i in doc:
        temp = i.getText()
        temp = temp.replace("\n", " ")
        ext_text = ext_text + temp
        temp = ""
    print("Text length : ", len(ext_text))

    # Find a word with a period and create a sentence by a regular expression.
    txt = ""
    final_sent = re.compile("[a-z]*\.")
    for i in range(len(to_text)):
        txt = txt + to_text[i]
        m = final_sent.findall(to_text[i])
        if m:
            txt = txt + "\n"

    # pdf to binary text 저장
    file = open('./data/ext_text.txt', 'wb')
    file.write(txt.encode('UTF-8'))
    file.close()

    # 저장된 TXT 파일 읽기
    file = open('./data/ext_text.txt', 'rb')
    text = file.readlines()
    file.close()

    print("문장 길이 : ", len(text))
    return t_text

# Papago API 번역 및 저장
def trans(id, secret, inputtext, line=10):
    text = ""
    for i in range(line):
        result_txt = papagoAPI(id, secret, inputtext[i])
        print("번역결과 {} : {}".format(i, result_txt))
        text = text + result_txt + "\n"
        time.sleep(2)

    trans_file = open("./data/trans_result.txt", 'w')
    trans_file.write(text)
    trans_file.close()

if __name__ == "__main__":
    # pdf to text 함수 호출
    eng_text = pdfToText('./data/test.pdf')
    # Naver 번역 API ID, Secret(password)
    id = "발급받은ClientID"
    secret = "발급받은ClientSecret"
    # Text 영문, Papago 번역후 Text파일저
    trans(id, secret, eng_text, line=2)
