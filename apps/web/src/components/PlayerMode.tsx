'use client';

import { useState, useRef, useEffect } from 'react';
import ChatInterface from './ChatInterface';
import VideoUploadArea from './VideoUploadArea';
import VideoFeedbackSection from './VideoFeedbackSection';
import { useGolfWebSocket } from '../hooks/useGolfWebSocket';
import { sendStopCommand } from '../services/mqtt.service';
import type { Message } from '../types';

export default function PlayerMode() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [feedbackHTML, setFeedbackHTML] = useState<string | null>(null);
  const [pendingTimestamp, setPendingTimestamp] = useState<string | null>(null);
  
  // WebSocket connection
  const { isConnected, subscribe, results, error: wsError } = useGolfWebSocket();
  const [chatMessages, setChatMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Welcome to SwingAI Lab! Ask me anything about your golf swing and I'll help you improve your form.",
      timestamp: new Date(),
    },
  ]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const handleVideoUpload = async (file: File) => {
    setVideoFile(file);
    setFeedbackHTML(null);
    setIsProcessingVideo(true);

    try {
      // Step 1: Upload video via API
      const response = await sendStopCommand(file);
      console.log('✓ Video uploaded. Full response:', JSON.stringify(response, null, 2));
      console.log('✓ Response keys:', Object.keys(response));
      console.log('✓ Response.timestamp:', response.timestamp);
      console.log('✓ Response type:', typeof response);

      // Step 2: Subscribe to WebSocket session if timestamp is available
      if (response.timestamp) {
        if (isConnected) {
          // Subscribe immediately if connected
          subscribe(response.timestamp);
        } else {
          // Store timestamp and wait for connection
          console.warn('⚠ WebSocket not connected. Will subscribe when connected...');
          setPendingTimestamp(response.timestamp);
        }
      } else {
        console.warn('⚠ No timestamp in response. Cannot subscribe to analysis.');
        setIsProcessingVideo(false);
      }
    } catch (error: any) {
      console.error('✗ Failed to upload video:', error);
      setIsProcessingVideo(false);
    }
  };

  // Handle WebSocket results
  useEffect(() => {
    if (results.length > 0) {
      // Combine all analysis HTML from multiple hits
      const combinedHTML = results
        .map((result, index) => {
          if (result.analysisHTML) {
            return `<div class="hit-analysis" data-hit-index="${result.hitIndex}">
              ${results.length > 1 ? `<h4>Hit #${result.hitIndex}</h4>` : ''}
              ${result.analysisHTML}
            </div>`;
          }
          return '';
        })
        .filter(Boolean)
        .join('<hr class="my-4" />');

      if (combinedHTML) {
        setFeedbackHTML(combinedHTML);
        setIsProcessingVideo(false);
      }
    }
  }, [results]);

  // Auto-subscribe when WebSocket connects and we have pending timestamp
  useEffect(() => {
    if (isConnected && pendingTimestamp) {
      console.log('✓ WebSocket connected. Subscribing to timestamp:', pendingTimestamp);
      subscribe(pendingTimestamp);
      setPendingTimestamp(null);
    }
  }, [isConnected, pendingTimestamp, subscribe]);

  // Handle WebSocket errors
  useEffect(() => {
    if (wsError) {
      console.error('WebSocket error:', wsError);
      // Don't stop processing, just log the error
    }
  }, [wsError]);

  const handleSendMessage = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    };
    setChatMessages((prev) => [...prev, userMessage]);

    setIsAnalyzing(true);

    // TODO: Call API for chat response - replace with real API
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'This is a mock response. API integration will be implemented later.',
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, assistantMessage]);
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <div className='flex min-h-[calc(100vh-80px)] flex-col gap-4 p-4 sm:gap-6 sm:p-6 bg-gradient-to-br from-white to-gray-50'>
      <div className='mx-auto w-full max-w-4xl'>
        <div className='mb-8'>
          <h1 className='text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent'>
            The AI Lab
          </h1>
          <p className='mt-2 text-gray-600'>Upload or record your swing and get instant AI-powered biomechanics analysis</p>
        </div>

        {/* Video Upload Section */}
        <VideoUploadArea 
          videoFile={videoFile} 
          onUpload={handleVideoUpload}
          onReset={() => setVideoFile(null)}
        />

        {/* Video Feedback Section - Top Half */}
        <div className="mb-4">
          <VideoFeedbackSection
            videoFileName={videoFile?.name || null}
            feedbackHTML={feedbackHTML}
            isProcessing={isProcessingVideo}
            isConnected={isConnected}
          />
        </div>

        {/* Chat Interface - Bottom Half */}
        <ChatInterface
          messages={chatMessages}
          isAnalyzing={isAnalyzing}
          onSendMessage={handleSendMessage}
          messagesEndRef={messagesEndRef}
        />
      </div>
    </div>
  );
}
