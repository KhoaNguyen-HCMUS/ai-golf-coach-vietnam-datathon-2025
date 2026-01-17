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
  const [videoData, setVideoData] = useState<Array<{
    base64: string;
    mimeType: string;
    filename: string;
    size: number;
  }>>([]);
  
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
      // Step 1: Generate timestamp FIRST
      const timestamp = Date.now().toString();
      console.log(`📅 Generated timestamp: ${timestamp}`);

      // Step 2: Subscribe BEFORE uploading
      if (isConnected) {
        subscribe(timestamp);
        console.log(`✓ Subscribed with timestamp: ${timestamp}`);
      } else {
        console.warn(`⚠ WebSocket not connected. Storing timestamp for later...`);
        setPendingTimestamp(timestamp);
      }

      // Step 3: Rename file with timestamp
      const renamedFile = new File(
        [file],
        `video_${timestamp}_${file.name}`,
        { type: file.type, lastModified: file.lastModified }
      );
      console.log(`✓ Renamed file to: ${renamedFile.name}`);

      // Step 4: Upload video with renamed file
      const response = await sendStopCommand(renamedFile);
      console.log(`✓ Video uploaded. Response timestamp:`, response.timestamp);
      console.log(`✓ Expected timestamp: ${timestamp}`);

      if (response.timestamp !== timestamp) {
        console.warn(`⚠ Timestamp mismatch! Response: ${response.timestamp}, Expected: ${timestamp}`);
      }
    } catch (error: any) {
      console.error('✗ Failed to upload video:', error);
      setIsProcessingVideo(false);
    }
  };

  // Handle WebSocket results
  useEffect(() => {
    console.log('🟣 [PlayerMode] Results changed:', results.length);
    console.log('🟣 [PlayerMode] Results data:', results);
    
    if (results.length > 0) {
      // Extract video data
      const videos = results
        .filter((result) => result && result.video)
        .map((result) => result.video!);
      setVideoData(videos);
      console.log('🟣 [PlayerMode] Extracted video data:', videos.length, 'videos');

      // Combine all analysis HTML from multiple hits
      const combinedHTML = results
        .map((result, index) => {
          console.log(`🟣 [PlayerMode] Processing result ${index}:`, {
            clipId: result?.clipId,
            hitIndex: result?.hitIndex,
            hasAnalysisHTML: !!result?.analysisHTML,
            analysisHTMLLength: result?.analysisHTML?.length || 0,
            hasVideo: !!result?.video
          });
          
          if (result && result.analysisHTML) {
            const html = `<div class="hit-analysis" data-hit-index="${result.hitIndex}">
              ${results.length > 1 ? `<h4>Hit #${result.hitIndex}</h4>` : ''}
              ${result.analysisHTML}
            </div>`;
            console.log(`🟣 [PlayerMode] Generated HTML for hit #${result.hitIndex}`);
            return html;
          }
          console.log(`🟡 [PlayerMode] Result ${index} has no analysisHTML`);
          return '';
        })
        .filter(Boolean)
        .join('<hr class="my-4" />');

      console.log('🟣 [PlayerMode] Combined HTML length:', combinedHTML.length);
      
      if (combinedHTML) {
        console.log('🟣 [PlayerMode] Setting feedbackHTML and stopping processing');
        setFeedbackHTML(combinedHTML);
        setIsProcessingVideo(false);
      } else {
        console.log('🟡 [PlayerMode] No combined HTML generated');
      }
    } else {
      console.log('🟡 [PlayerMode] No results yet');
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
            videoData={videoData}
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
