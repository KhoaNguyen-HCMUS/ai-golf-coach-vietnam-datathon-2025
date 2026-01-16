#pragma once
#include <WiFi.h>
#include <PubSubClient.h>
#include <WiFiClientSecure.h>

class MqttClient {
public:
    void begin();
    void loop();

    bool shouldStart();
    bool shouldStop();
    void resetFlags();

    void publishSession(unsigned long startTime,
                        const unsigned long* hits,
                        int count);

private:
    WiFiClient wifi;
    PubSubClient mqtt;

    bool startFlag = false;
    bool stopFlag  = false;

    static MqttClient* instance;
    static void callback(char* topic, byte* payload, unsigned int length);
};
