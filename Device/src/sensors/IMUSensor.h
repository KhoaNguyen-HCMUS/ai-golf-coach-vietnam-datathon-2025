#pragma once

#include <MPU6050.h>

class IMUSensor {
public:
    IMUSensor();

    bool begin();
    void read();

    float getAccMagnitude();
    float getGyroMagnitude();

private:
    MPU6050 mpu;

    float accMag;
    float gyroMag;
};
