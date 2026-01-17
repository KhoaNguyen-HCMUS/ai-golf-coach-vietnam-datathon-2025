import { Router } from 'express';
import multer from 'multer'
import * as mqttController from '../controllers/mqtt.controller.js'

export const router = Router()

const storage = multer.diskStorage({
    destination: './videos/Input/',
    filename: (req, file, cb) => {
        cb(null, `video_${Date.now()}_${file.originalname}`)
    }
})

const upload = multer({ 
    storage,
    limits: { fileSize: 500 * 1024 * 1024 } // 500MB max
})

router.post('/cmd/start', mqttController.startController)
router.post('/cmd/stop', upload.single('video'), mqttController.stopController)
router.post('/cmd/stop-no-video', mqttController.stopWithoutVideoController)

export default router;