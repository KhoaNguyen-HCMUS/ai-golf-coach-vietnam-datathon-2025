#include "IMUSensor.h"
#include <Wire.h>
#include <math.h>

IMUSensor::IMUSensor()
    : accMag(0.0f), gyroMag(0.0f) {}

bool IMUSensor::begin() {
    Wire.begin();
    mpu.initialize();
    return mpu.testConnection();
}

void IMUSensor::read() {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    float accX = ax / 16384.0f;
    float accY = ay / 16384.0f;
    float accZ = az / 16384.0f;

    float gyroX = gx / 131.0f;
    float gyroY = gy / 131.0f;
    float gyroZ = gz / 131.0f;

    accMag = sqrt(accX * accX + accY * accY + accZ * accZ);
    gyroMag = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);
}

float IMUSensor::getAccMagnitude() {
    return accMag;
}

float IMUSensor::getGyroMagnitude() {
    return gyroMag;
}
