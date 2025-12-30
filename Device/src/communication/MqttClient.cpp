#include "MqttClient.h"
#include "config.h"
#include <ArduinoJson.h>

MqttClient* MqttClient::instance = nullptr;

void MqttClient::begin() {
    instance = this;

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }

    mqttClient.setClient(wifiClient);
    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback(callback);

    while (!mqttClient.connected()) {
        mqttClient.connect("ImpactSensorNode");
        delay(500);
    }

    mqttClient.subscribe(MQTT_TOPIC_CMD);
}

void MqttClient::loop() {
    mqttClient.loop();
}

void MqttClient::callback(char* topic, byte* payload, unsigned int length) {
    if (!instance) return;

    String msg;
    for (unsigned int i = 0; i < length; i++) {
        msg += (char)payload[i];
    }

    if (String(topic) == MQTT_TOPIC_CMD && msg == "START") {
        instance->startMeasureFlag = true;
    }
}

bool MqttClient::shouldStartMeasuring() {
    return startMeasureFlag;
}

void MqttClient::resetCommand() {
    startMeasureFlag = false;
}

void MqttClient::publishImpact(unsigned long startTimestamp,
                               unsigned long impactTimestamp,
                               float accMag,
                               float gyroMag) {
    StaticJsonDocument<256> doc;

    doc["impact"] = true;
    doc["start_time"] = startTimestamp;
    doc["impact_time"] = impactTimestamp;
    doc["delay_ms"] = impactTimestamp - startTimestamp;
    doc["acc_mag"] = accMag;
    doc["gyro_mag"] = gyroMag;

    char buffer[256];
    serializeJson(doc, buffer);

    mqttClient.publish(MQTT_TOPIC_RESULT, buffer);
}
