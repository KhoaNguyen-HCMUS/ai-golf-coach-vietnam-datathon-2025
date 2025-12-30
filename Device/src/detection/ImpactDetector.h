#pragma once

class ImpactDetector {
public:
    ImpactDetector(float accTh, float gyroTh, unsigned long cooldownMs);

    bool detect(float accMag, float gyroMag);

private:
    float accThreshold;
    float gyroThreshold;
    unsigned long cooldown;
    unsigned long lastImpactTime;
};
