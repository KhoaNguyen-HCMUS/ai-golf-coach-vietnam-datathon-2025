import { WebSocketServer } from 'ws'

const clients = new Map() // timestamp -> [ws, ws, ...]

export const initWebSocket = (server) => {
    const wss = new WebSocketServer({ server })
    
    wss.on('connection', (ws) => {
        console.log('[WS] Client connected')
        
        ws.on('message', (msg) => {
            try {
                const data = JSON.parse(msg)
                
                if (data.type === 'subscribe') {
                    if (!clients.has(data.timestamp)) {
                        clients.set(data.timestamp, [])
                    }
                    clients.get(data.timestamp).push(ws)
                    
                    ws.send(JSON.stringify({
                        type: 'subscribed',
                        timestamp: data.timestamp
                    }))
                    
                    console.log(`[WS] Subscribed to ${data.timestamp}`)
                }
            } catch (e) {
                console.error('[WS] Error:', e)
            }
        })
        
        ws.on('close', () => {
            clients.forEach((list) => {
                const idx = list.indexOf(ws)
                if (idx > -1) list.splice(idx, 1)
            })
            console.log('[WS] Client disconnected')
        })
    })
    
    console.log('[WebSocket] Ready')
}

export const sendToClient = (timestamp, message) => {
    const list = clients.get(timestamp)
    if (!list || list.length === 0) return
    
    const msg = JSON.stringify(message)
    list.forEach(ws => {
        if (ws.readyState === 1) {
            ws.send(msg)
        }
    })
    
    console.log(`[WS] Sent to ${timestamp}: ${list.length} clients`)
}