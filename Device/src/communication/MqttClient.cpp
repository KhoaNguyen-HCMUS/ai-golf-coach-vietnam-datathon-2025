#include "MqttClient.h"
#include "../config.h"
#include <ArduinoJson.h>


MqttClient* MqttClient::instance = nullptr;

void MqttClient::begin() {
    instance = this;

    WiFi.mode(WIFI_STA);  // Thêm dòng này
    WiFi.disconnect();     // Clear previous connections
    delay(100);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.println("SSID: " + String(WIFI_SSID) + " - " + "PASSWORD: " + String(WIFI_PASSWORD));
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) 
    {
        Serial.print(".");
        delay(300);
    }
    Serial.println();
    Serial.println("Connected to Wifi");

    WiFiClientSecure* secureClient = new WiFiClientSecure();
    secureClient->setInsecure(); 

    mqtt.setClient(*secureClient);
    mqtt.setServer(MQTT_BROKER, MQTT_PORT);
    mqtt.setCallback(callback);
    
    while (!mqtt.connected()) {
        mqtt.connect(MQTT_CLIENT_ID, MQTT_USER, MQTT_PASS);
        delay(300);
    }
    Serial.println("Connected with MQTT");
    mqtt.subscribe(MQTT_TOPIC_CMD);
    Serial.println("Subscribed to topic " + String(MQTT_TOPIC_CMD));
}

void MqttClient::loop() {
    mqtt.loop();
}

void MqttClient::callback(char* topic, byte* payload, unsigned int length) {
    if (!instance) return;

    String msg;
    for (unsigned int i = 0; i < length; i++)
        msg += (char)payload[i];

    if (msg == "START") instance->startFlag = true;
    if (msg == "STOP")  instance->stopFlag  = true;
}

bool MqttClient::shouldStart() { return startFlag; }
bool MqttClient::shouldStop()  { return stopFlag; }

void MqttClient::resetFlags() {
    startFlag = false;
    stopFlag  = false;
}

void MqttClient::publishSession(unsigned long startTime,
                                const unsigned long* hits,
                                int count) {
    StaticJsonDocument<512> doc;
    doc["start_time"] = startTime;

    JsonArray arr = doc.createNestedArray("hits");
    for (int i = 0; i < count; i++)
        arr.add(hits[i] - startTime);

    char buf[512];
    serializeJson(doc, buf);
    mqtt.publish(MQTT_TOPIC_RESULT, buf);
}
