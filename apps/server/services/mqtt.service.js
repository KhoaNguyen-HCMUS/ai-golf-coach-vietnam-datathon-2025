import mqtt from 'mqtt'
import { MQTT_CONFIG, MQTT_TOPICS } from '../config/config.js'

let client = null
let lastHits = null

export const connectMqtt = () => {
    client = mqtt.connect(MQTT_CONFIG.broker, {
        port: MQTT_CONFIG.portMqtt,
        username: MQTT_CONFIG.username,
        password: MQTT_CONFIG.password,
        rejectUnauthorized: false,
    })

    client.on('connect', () => {
        console.log('Connected to MQTT broker')
        client.subscribe(MQTT_TOPICS.result, (err) => {
            if (err) {
                console.error('Failed to subscribe to topic:', MQTT_TOPICS.result)
            } else {
                console.log('Subscribed to topic:', MQTT_TOPICS.result) 
            }
        })

        client.on('disconnect', () => {
            console.log('Disconnected from MQTT broker')
        })

        client.on('error', (error) => {
            console.error('MQTT Error:', error)
            throw new Error('MQTT connection error')
        })

        client.on('message', (topic, message) => {
            if (topic === MQTT_TOPICS.result) {
                const data = JSON.parse(message.toString())
                console.log("Result from esp32: ", data)
                lastHits = data.hits || null
            }
        })
    })
}


export const publishCmd = async (cmd) => {
    if (!client || !client.connected){
        throw new Error('MQTT client is not connected')
    }

    return new Promise((resolve, reject) => {
        client.publish(MQTT_TOPICS.cmd, cmd, { qos: 1 }, (err) => {
            if (err) reject(new Error('Publish failed: ' + err.message))
            else resolve()
        })
    })
}

export const getLastHits = () => {
    const data = lastHits
    lastHits = null
    return data
}
