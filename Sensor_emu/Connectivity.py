# Connectivity

import time
import paho.mqtt.client as mqtt


## MQTT Parameters
hostname = "Python_Sensor"
broker_port = 1883
topic = "80"
#broker_address = "192.168.1.101"
broker_address = "192.168.1.104"


def MQTTPublish(data):
      
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=hostname)
    
    client.connect(broker_address, broker_port)

    for v in data:
        result = client.publish(topic, v)

        while result[0] != 0:
            time.sleep(1)
            result = client.publish(topic, v) 

        time.sleep(1)
    
    client.disconnect()
    

def MQTTConection(get_queue=False):

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc==0:
            print("Connected OK Returned code=",rc)
        else:
            print("Bad connection Returned code=",rc)
        client.subscribe(topic)
    
    def on_message(client, userdata, msg):
        userdata.append(msg.payload.decode())
        if len(userdata) == 60:
            get_queue.put(userdata)
            print("*********************************************************")
            userdata.clear()
        
        #print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        
        

    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=hostname)
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.user_data_set([])

    client.connect(broker_address, broker_port)
    client.loop_forever()

    #return client

def MQTTDisconnection(client):
    client.loop_stop()
    client.disconnect()
