import { Router } from 'express';
import * as mqttController from '../controllers/mqtt.controller.js'

export const router = Router()

router.post('/cmd/start', mqttController.startController)
router.post('/cmd/stop', mqttController.stopController)

export default router;