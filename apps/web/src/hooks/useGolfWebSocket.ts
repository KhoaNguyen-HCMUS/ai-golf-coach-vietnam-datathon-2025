import { useEffect, useRef, useState, useCallback } from 'react';

interface WebSocketMessage {
  type: 'completed' | 'error' | 'status';
  data?: {
    clipId: string;
    timestamp: string;
    hitIndex: number;
    analysisHTML: string;
    video?: {
      base64: string;
      mimeType: string;
      filename: string;
      size: number;
    };
  };
  error?: string;
}

interface UseGolfWebSocketReturn {
  isConnected: boolean;
  subscribe: (timestamp: string) => void;
  results: WebSocketMessage['data'][];
  error: string | null;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://10.20.121.231:5050';

export function useGolfWebSocket(): UseGolfWebSocketReturn {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [results, setResults] = useState<WebSocketMessage['data'][]>([]);
  const [error, setError] = useState<string | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(WS_URL);

      ws.current.onopen = () => {
        console.log('✓ Connected to AI Golf Coach WebSocket');
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      ws.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('✓ Received WebSocket message:', message);

          if (message.type === 'completed' && message.data) {
            setResults((prev) => {
              // Check if this result already exists (avoid duplicates)
              const newData = message.data!;
              const exists = prev.some((r) => r && r.clipId === newData.clipId);
              if (exists) {
                return prev;
              }
              return [...prev, newData];
            });
            console.log(`✓ Analysis received for hit #${message.data.hitIndex}`);
          } else if (message.type === 'error') {
            setError(message.error || 'Unknown error from server');
            console.error('✗ WebSocket error message:', message.error);
          }
        } catch (err) {
          console.error('✗ Failed to parse WebSocket message:', err);
          setError('Failed to parse server message');
        }
      };

      ws.current.onerror = (err) => {
        console.error('✗ WebSocket error:', err);
        setError('WebSocket connection error');
      };

      ws.current.onclose = () => {
        console.log('✗ Disconnected from WebSocket');
        setIsConnected(false);

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          const delay = Math.pow(2, reconnectAttempts.current) * 1000; // Exponential backoff
          console.log(`Reconnecting in ${delay}ms... (attempt ${reconnectAttempts.current})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          setError('Max reconnection attempts reached');
        }
      };
    } catch (err) {
      console.error('✗ Failed to create WebSocket:', err);
      setError('Failed to connect to WebSocket server');
    }
  }, []);

  const subscribe = useCallback((timestamp: string) => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      console.error('✗ WebSocket is not connected. Cannot subscribe.');
      setError('WebSocket is not connected');
      return;
    }

    const subscribeMessage = {
      type: 'subscribe',
      timestamp: timestamp,
    };

    ws.current.send(JSON.stringify(subscribeMessage));
    console.log('✓ Subscribed to session:', timestamp);
    setResults([]); // Clear previous results when subscribing to new session
  }, []);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connect]);

  return {
    isConnected,
    subscribe,
    results,
    error,
  };
}
