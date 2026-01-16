import * as mqttService from '../services/mqtt.service.js';
export const startController = async (req, res) => {
    try {
        await mqttService.publishCmd('START')
        res.status(200).json({ message: 'START command published' });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
}

export const stopController = async (req, res) => {
    try {
        await mqttService.publishCmd('STOP')
        res.status(200).json({ message: 'STOP command published' });
    }   catch (error) {
        res.status(500).json({ error: error.message });
    }
}