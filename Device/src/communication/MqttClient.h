#pragma once

#include <WiFi.h>
#include <PubSubClient.h>

class MqttClient {
public:
    void begin();
    void loop();

    bool shouldStartMeasuring();
    void resetCommand();

    void publishImpact(unsigned long startTimestamp,
                       unsigned long impactTimestamp,
                       float accMag,
                       float gyroMag);

private:
    WiFiClient wifiClient;
    PubSubClient mqttClient;

    bool startMeasureFlag = false;

    static void callback(char* topic, byte* payload, unsigned int length);
    static MqttClient* instance;
};
