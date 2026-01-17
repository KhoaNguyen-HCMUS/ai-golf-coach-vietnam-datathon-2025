import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import http from 'http'
import * as mqttService from './services/mqtt.service.js'
import * as websocketService from './services/websocket.service.js'
import mqttRouter from './routes/mqtt.route.js'

const app = express()
const PORT = process.env.PORT
const HOST = process.env.HOST

app.use(cors())
app.use(express.json())
app.use('/api/mqtt', mqttRouter)

app.get('/', (req, res) => {
  res.json({ message: 'AI Golf Coach API' })
})

// Tạo HTTP server cho WebSocket
const server = http.createServer(app)

server.listen(PORT, HOST, () => {
  console.log(`Server: http://${HOST}:${PORT}`)
  mqttService.connectMqtt()
  websocketService.initWebSocket(server)
  console.log('All services ready')
})