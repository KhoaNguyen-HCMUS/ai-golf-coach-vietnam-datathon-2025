#include <Arduino.h>
#include "config.h"
#include "communication/MqttClient.h"
#include "detection/ImpactDetector.h"

#define MAX_HITS 32

MqttClient mqtt;
ImpactDetector* detector;

unsigned long sessionStart = 0;
unsigned long hits[MAX_HITS];
int hitCount = 0;
bool running = false;

extern ImpactDetector* createImpactDetector();

void setup() {
    Serial.begin(115200);

    detector = createImpactDetector();
    detector->begin();

    mqtt.begin();
    Serial.println("ESP32 READY");
}

void loop() {
    mqtt.loop();

    if (mqtt.shouldStart()) {
        running = true;
        hitCount = 0;
        sessionStart = millis();
        mqtt.resetFlags();
        Serial.println("START");
    }

    if (running && detector->detect()) {
        if (hitCount < MAX_HITS) {
            hits[hitCount++] = millis();
            Serial.println("HIT at " + String(hits[hitCount - 1]));
        }
    }

    if (mqtt.shouldStop()) {
        running = false;
        mqtt.publishSession(sessionStart, hits, hitCount);
        mqtt.resetFlags();
        Serial.println("STOP");
    }
}
