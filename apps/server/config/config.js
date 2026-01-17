export const MQTT_CONFIG = {
    broker: process.env.MQTT_BROKER,
    portMqtt: process.env.MQTT_PORT,
    username: process.env.MQTT_USERNAME,
    password: process.env.MQTT_PASSWORD,
}

export const MQTT_TOPICS = {
    cmd: process.env.MQTT_TOPIC_CMD,
    result: process.env.MQTT_TOPIC_RESULT,
}