from Internet import *

loopFlag = 1

def printMenu():
    print("**********************************************")
    print("             실시간 항공운항 서비스              ")
    print("**********************************************")
    print("             Menu                             ")
    print("             시간대별 국제 항공편 검색 :s         ")
    print("             시간대별 국내 항공편 검색 :d         ")
    print("             공항별   국내선 검색  :h            ")
    print("             종합 날씨 조회 :     r             ")
    print("             METAR 조회:          w            ")
    print("             면세점 시설 검색:      f           ")
    print("             비면세점 구역 매장 검색:n            ")
    print("             프로그램 종료:       q             ")
    print("**********************************************")


def QuitBookMgr():
    global loopFlag
    loopFlag=0


def launcherFunction(menu):
    if menu =='s':
        inter_airline()
    elif menu =='d':
        dom_airline()
    elif menu == 'q':
        QuitBookMgr()
    elif menu == 'w':
        weather()
    elif menu == 'h':
        airportDomainSearch()
    elif menu == 'r':
        generalWeatherCondition()
    elif menu =='f':
        searchDutyFacilities()
    elif menu =='n':
        searchNodutyFacilities()
    else:
        print("유효하지 않은 메뉴 키입니다.")


while (loopFlag > 0):
    printMenu()
    menuKey = str(input('select menu :'))
    launcherFunction(menuKey)

else:
    print("프로그램 종료합니다.")