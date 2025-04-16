# Nama : I Gusti Lanang Ari Trisne Sudana
# Nim : 13219046
# Tugas WPC day 3

import paho.mqtt.client as mqtt #import the client1
import time
import random
broker_address="io.adafruit.com"
clientId="Lanang" #ganti sesuai nama kalian
username= "lananggusti66" #Ganti sesuai username kalian
password= "aio_aLeS41joWYyu1gQgtDDgPxlH32pB" #ganti sesuai token API kalian

#broker_address="iot.eclipse.org" #use external broker
client = mqtt.Client(clientId) #create new instance
client.username_pw_set(username, password)
client.connect(broker_address) #connect to broker
suhu=0
cahaya=0
kelembaban=0
led=0
while True:
	#update nilai
	suhu=random.randint(25, 35)
	cahaya=random.randint(0, 100)/100
	kelembaban=random.randint(7500, 9000)/100
	led=random.randint(0, 1)


    #publish
	client.publish(username+"/feeds/"+"sensorsuhu",suhu)
	client.publish(username+"/feeds/"+"sensorcahaya",cahaya)
	client.publish(username+"/feeds/"+"sensorkelembaban",kelembaban)
	client.publish(username+"/feeds/"+"led",led)

	time.sleep(10)