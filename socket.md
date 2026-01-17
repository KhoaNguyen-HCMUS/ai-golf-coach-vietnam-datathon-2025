# WebSocket Setup Guide - AI Golf Coach Frontend

## Server Information

- **WebSocket URL**: `ws://10.20.121.231:5050`
- **Connection Type**: WebSocket (ws://)
- **Real-time**: Bidirectional communication for live swing analysis

---

## 1. Connection Setup

### Basic Connection

```javascript
// Create WebSocket connection
const ws = new WebSocket('ws://10.20.121.231:5050');

// Handle connection open
ws.onopen = () => {
  console.log('✓ Connected to AI Golf Coach server');
};

// Handle errors
ws.onerror = (error) => {
  console.error('✗ WebSocket Error:', error);
};

// Handle connection close
ws.onclose = () => {
  console.log('✗ Disconnected from server');
};
```

### React Hook Example

```javascript
import { useEffect, useRef, useState } from 'react';

export function useGolfWebSocket() {
  const ws = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Connect to WebSocket
    ws.current = new WebSocket('ws://10.20.121.231:5050');

    ws.current.onopen = () => {
      console.log('✓ Connected');
      setIsConnected(true);
    };

    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        setData(message);
        console.log('✓ Received analysis:', message);
      } catch (err) {
        console.error('Error parsing message:', err);
      }
    };

    ws.current.onerror = (error) => {
      console.error('✗ WebSocket error:', error);
      setError(error);
    };

    ws.current.onclose = () => {
      console.log('✗ Disconnected');
      setIsConnected(false);
    };

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  return { ws: ws.current, isConnected, data, error };
}
```

---

## 2. Subscribe to Session (Important!)

After uploading a video and receiving a **timestamp response**, you MUST subscribe to that session to receive analysis results.

### When to Subscribe

1. User uploads video via `/api/mqtt/cmd/stop` endpoint
2. Server returns response with `timestamp`
3. **Immediately subscribe** to that timestamp via WebSocket
4. Wait for analysis results (takes 5-10 seconds)

### Subscription Message Format

```javascript
const timestamp = "1705424890123"; // From upload response

const subscribeMessage = {
  type: 'subscribe',
  timestamp: timestamp
};

ws.send(JSON.stringify(subscribeMessage));
console.log('✓ Subscribed to session:', timestamp);
```

### Complete Example: Upload → Subscribe → Receive

```javascript
async function uploadAndAnalyze(videoFile) {
  try {
    // Step 1: Upload video
    const formData = new FormData();
    formData.append('video', videoFile);

    const uploadResponse = await fetch('http://10.20.121.231:5050/api/mqtt/cmd/stop', {
      method: 'POST',
      body: formData
    });

    const uploadData = await uploadResponse.json();
    const sessionTimestamp = uploadData.timestamp;

    console.log('✓ Video uploaded. Timestamp:', sessionTimestamp);

    // Step 2: Subscribe to session results
    const subscribeMsg = {
      type: 'subscribe',
      timestamp: sessionTimestamp
    };

    ws.send(JSON.stringify(subscribeMsg));
    console.log('✓ Subscribed to analysis');

    // Step 3: Wait for analysis (onmessage will handle it)
    // See "Receiving Messages" section below

  } catch (error) {
    console.error('Upload failed:', error);
  }
}
```

---

## 3. Receiving Messages

### Message Structure

When analysis is complete, you'll receive:

```javascript
{
  type: 'completed',
  data: {
    clipId: 'timestamp_hit_1',
    timestamp: '1705424890123',
    hitIndex: 1,
    analysisHTML: '<h3>Overall Assessment</h3><p>Score: 85/100...</p>...',
    video: {
      base64: 'AAAAHGZ0eXBpc29tAAACAGlzb21pc28yYWM4...',
      mimeType: 'video/mp4',
      filename: 'hit_1.mp4',
      size: 1245678
    }
  }
}
```

### Message Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `'completed'` for analysis results |
| `clipId` | string | Unique identifier for this swing clip |
| `timestamp` | string | Session timestamp (matches your subscription) |
| `hitIndex` | number | Swing number (1st, 2nd, 3rd hit, etc.) |
| `analysisHTML` | string | **HTML feedback** from AI coach (ready to display) |
| `video.base64` | string | Video file encoded in base64 |
| `video.mimeType` | string | Always `'video/mp4'` |
| `video.filename` | string | Original filename |
| `video.size` | number | File size in bytes |

---

## 4. Display Analysis HTML

The `analysisHTML` is ready-to-display HTML with professional coaching feedback:

### React Component Example

```javascript
import { useGolfWebSocket } from './useGolfWebSocket';

export function AnalysisDisplay() {
  const { data } = useGolfWebSocket();

  if (!data || data.type !== 'completed') {
    return <p>Waiting for analysis...</p>;
  }

  return (
    <div className="analysis-container">
      <h2>Hit #{data.data.hitIndex} Analysis</h2>
      
      {/* Display HTML analysis */}
      <div
        className="coaching-feedback"
        dangerouslySetInnerHTML={{ __html: data.data.analysisHTML }}
      />

      {/* Display video */}
      <div className="video-player">
        <video width="400" controls>
          <source
            src={`data:${data.data.video.mimeType};base64,${data.data.video.base64}`}
            type={data.data.video.mimeType}
          />
          Your browser doesn't support HTML5 video
        </video>
      </div>
    </div>
  );
}
```

### HTML Structure from AI Coach

The returned `analysisHTML` contains sections like:

```html
<h3>Overall Assessment</h3>
<p>Score: <strong>85/100</strong></p>

<h3>Technical Analysis</h3>
<ul>
  <li style="color: #4caf50;">Excellent stance positioning</li>
  <li style="color: #ff9800;">Hip rotation could be improved</li>
</ul>

<h3>Key Strengths</h3>
<ul>
  <li>Smooth backswing motion</li>
  <li>Good impact position</li>
</ul>

<h3>Areas for Improvement</h3>
<ul>
  <li style="color: #f44336;">Follow-through inconsistent</li>
</ul>

<h3>Specific Recommendations</h3>
<ol>
  <li>Practice follow-through drills</li>
  <li>Work on hip rotation flexibility</li>
</ol>
```

**Color Coding**:
- 🟢 **Green (#4caf50)**: Good technique
- 🟠 **Orange (#ff9800)**: Areas to improve
- 🔴 **Red (#f44336)**: Critical issues

---

## 5. Handle Video Playback

### Display Video from Base64

```javascript
function VideoPlayer({ videoBase64, mimeType }) {
  return (
    <video width="640" height="480" controls>
      <source
        src={`data:${mimeType};base64,${videoBase64}`}
        type={mimeType}
      />
      Your browser doesn't support HTML5 video
    </video>
  );
}

// Usage
<VideoPlayer
  videoBase64={data.data.video.base64}
  mimeType={data.data.video.video}
/>
```

### Performance Note
- Videos are base64 encoded for direct embedding
- Typical size: 1-2 MB per clip
- For larger videos or better performance, see server `/videos` route

---

## 6. Multiple Hits Handling

When a session has multiple swings, you'll receive **multiple messages**:

```javascript
export function MultiHitAnalysis() {
  const { data } = useGolfWebSocket();
  const [allResults, setAllResults] = useState([]);

  useEffect(() => {
    if (data?.type === 'completed') {
      // Add new result (each hit sends separate message)
      setAllResults(prev => [...prev, data.data]);
      console.log(`✓ Received hit #${data.data.hitIndex}`);
    }
  }, [data]);

  return (
    <div>
      <h1>Session Analysis ({allResults.length} swings)</h1>
      {allResults.map((result) => (
        <div key={result.clipId} className="hit-result">
          <h2>Hit #{result.hitIndex}</h2>
          <div dangerouslySetInnerHTML={{ __html: result.analysisHTML }} />
          <VideoPlayer {...result.video} />
        </div>
      ))}
    </div>
  );
}
```

---

## 7. Error Handling

### Connection Loss

```javascript
function setupAutoReconnect(ws, maxRetries = 5) {
  let retries = 0;

  ws.onclose = () => {
    if (retries < maxRetries) {
      retries++;
      const delay = Math.pow(2, retries) * 1000; // Exponential backoff
      console.log(`Reconnecting in ${delay}ms... (attempt ${retries})`);
      setTimeout(() => {
        ws = new WebSocket('ws://10.20.121.231:5050');
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
    }
  };
}
```

### Invalid Messages

```javascript
ws.onmessage = (event) => {
  try {
    const message = JSON.parse(event.data);

    if (message.type === 'completed') {
      console.log('✓ Analysis received');
      handleAnalysis(message.data);
    } else {
      console.warn('⚠ Unknown message type:', message.type);
    }
  } catch (error) {
    console.error('✗ Failed to parse message:', event.data);
  }
};
```

---

## 8. Complete React Implementation

```javascript
import React, { useEffect, useState, useRef } from 'react';

export default function GolfCoachApp() {
  const ws = useRef(null);
  const [connected, setConnected] = useState(false);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Initialize WebSocket
    ws.current = new WebSocket('ws://10.20.121.231:5050');

    ws.current.onopen = () => {
      console.log('✓ Connected to AI Golf Coach');
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'completed') {
          setResults(prev => [...prev, message.data]);
          setLoading(false);
          console.log(`✓ Analysis received for hit #${message.data.hitIndex}`);
        }
      } catch (error) {
        console.error('Parse error:', error);
      }
    };

    ws.current.onerror = (error) => {
      console.error('✗ WebSocket error:', error);
    };

    ws.current.onclose = () => {
      console.log('✗ Disconnected');
      setConnected(false);
    };

    return () => ws.current?.close();
  }, []);

  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setResults([]);

    try {
      // Upload video
      const formData = new FormData();
      formData.append('video', file);

      const uploadRes = await fetch('http://10.20.121.231:5050/api/mqtt/cmd/stop', {
        method: 'POST',
        body: formData
      });

      const uploadData = await uploadRes.json();
      const timestamp = uploadData.timestamp;

      console.log('✓ Video uploaded:', timestamp);

      // Subscribe to results
      ws.current.send(JSON.stringify({
        type: 'subscribe',
        timestamp: timestamp
      }));

      console.log('✓ Subscribed to analysis');
    } catch (error) {
      console.error('Upload error:', error);
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>AI Golf Coach</h1>

      <div>
        <input
          type="file"
          accept="video/*"
          onChange={handleVideoUpload}
          disabled={!connected || loading}
        />
        <span>{connected ? '🟢 Connected' : '🔴 Disconnected'}</span>
        {loading && <span>⏳ Analyzing...</span>}
      </div>

      <div className="results">
        {results.map(result => (
          <div key={result.clipId} className="result-card">
            <h2>Hit #{result.hitIndex}</h2>

            <div className="analysis" dangerouslySetInnerHTML={{ __html: result.analysisHTML }} />

            <video width="400" controls>
              <source
                src={`data:${result.video.mimeType};base64,${result.video.base64}`}
                type={result.video.mimeType}
              />
            </video>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| **Connection refused** | Check server is running: `pnpm be dev` |
| **"Connection lost"** | Server may have crashed - restart backend |
| **No message received** | Ensure you **subscribed** with correct timestamp |
| **Message parse error** | Check browser console for invalid JSON |
| **Base64 video won't play** | Verify mimeType is `'video/mp4'` |
| **HTML not rendering** | Use `dangerouslySetInnerHTML` in React |

---

## 10. Quick Reference

### Subscribe Message
```json
{
  "type": "subscribe",
  "timestamp": "1705424890123"
}
```

### Analysis Response
```json
{
  "type": "completed",
  "data": {
    "clipId": "1705424890123_hit_1",
    "timestamp": "1705424890123",
    "hitIndex": 1,
    "analysisHTML": "<h3>Overall Assessment</h3>...",
    "video": {
      "base64": "AAAAHGZ0eXBp...",
      "mimeType": "video/mp4",
      "filename": "hit_1.mp4",
      "size": 1245678
    }
  }
}
```

### Key Endpoints
- **WebSocket**: `ws://10.20.121.231:5050`
- **Video Upload**: `POST http://10.20.121.231:5050/api/mqtt/cmd/stop`
- **Response**: `{ timestamp, outputDir, totalClips, segments }`

---

## Support

For issues or questions:
1. Check server logs: `pnpm be dev`
2. Check browser console (F12)
3. Verify network in DevTools
4. Ensure video upload completed successfully before subscribing
