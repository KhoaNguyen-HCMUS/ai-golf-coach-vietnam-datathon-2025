#include "ImpactDetector.h"
#include "../config.h"
#include <Arduino.h>

class ButtonImpactDetector : public ImpactDetector {
public:
    ButtonImpactDetector()
        : lastHitTime(0) {}

    void begin() override {
        pinMode(buttonPin, INPUT_PULLUP);
    }

    bool detect() override {
        unsigned long now = millis();
        if (digitalRead(buttonPin ) == LOW &&
            (now - lastHitTime) > 500) {
            lastHitTime = now;
            return true;
        }
        return false;
    }

private:
    unsigned long lastHitTime;
};

ImpactDetector* createImpactDetector() {
    static ButtonImpactDetector detector;
    return &detector;
}
