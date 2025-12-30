#include <Arduino.h>
#include "config.h"

#include "sensors/IMUSensor.h"
#include "detection/ImpactDetector.h"
#include "communication/MqttClient.h"

IMUSensor imu;
ImpactDetector detector(ACC_THRESHOLD, GYRO_THRESHOLD, COOLDOWN_MS);
MqttClient mqtt;

bool measuring = false;
unsigned long measureStartTime = 0;

void setup() {
    Serial.begin(115200);

    if (!imu.begin()) {
        Serial.println("IMU not detected");
        while (1);
    }

    mqtt.begin();
    Serial.println("Impact Sensor Node Ready");
}

void loop() {
    mqtt.loop();

    // Nhận lệnh START từ Web
    if (mqtt.shouldStartMeasuring() && !measuring) {
        measuring = true;
        measureStartTime = millis();
        mqtt.resetCommand();

        Serial.println("▶ Start measuring");
    }

    //  Chỉ đo khi có lệnh
    if (measuring) {
        imu.read();

        float accMag  = imu.getAccMagnitude();
        float gyroMag = imu.getGyroMagnitude();

        if (detector.detect(accMag, gyroMag)) {
            unsigned long impactTime = millis();

            mqtt.publishImpact(
                measureStartTime,
                impactTime,
                accMag,
                gyroMag
            );

            measuring = false;
            Serial.println("🎯 Impact detected & sent");
        }

        delay(SAMPLE_DELAY_MS);
    }
}
