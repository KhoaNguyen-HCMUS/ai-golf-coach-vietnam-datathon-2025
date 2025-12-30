#include "ImpactDetector.h"
#include <Arduino.h>

ImpactDetector::ImpactDetector(float accTh,
                               float gyroTh,
                               unsigned long cooldownMs)
    : accThreshold(accTh),
      gyroThreshold(gyroTh),
      cooldown(cooldownMs),
      lastImpactTime(0) {}

bool ImpactDetector::detect(float accMag, float gyroMag) {
    unsigned long now = millis();

    if (accMag > accThreshold &&
        gyroMag > gyroThreshold &&
        (now - lastImpactTime) > cooldown) {

        lastImpactTime = now;
        return true;
    }
    return false;
}
