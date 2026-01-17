#pragma once

class ImpactDetector {
public:
    virtual void begin() = 0;
    virtual bool detect() = 0;
};
