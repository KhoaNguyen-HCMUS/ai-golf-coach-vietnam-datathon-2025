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
    console.log(`[WS] Trying to send to timestamp: ${timestamp}`)
    console.log(`[WS] Available timestamps:`, Array.from(clients.keys()))
    
    const list = clients.get(timestamp)
    console.log(`[WS] Found ${list?.length || 0} clients for ${timestamp}`)
    
    if (!list || list.length === 0) {
        console.log(`[WS] NO CLIENTS for timestamp ${timestamp}`)
        return
    }
    
    const msg = JSON.stringify(message)
    let sentCount = 0
    list.forEach(ws => {
        if (ws.readyState === 1) {
            ws.send(msg)
            sentCount++
        }
    })
    
    console.log(`[WS] Sent to ${timestamp}: ${sentCount}/${list.length} clients`)
}