'use client';

import { useState, useRef, useEffect } from 'react';
import ChatInterface from './ChatInterface';
import VideoUploadArea from './VideoUploadArea';
import VideoFeedbackSection from './VideoFeedbackSection';
import type { Message } from '../types';

export default function PlayerMode() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Welcome to SwingAI Lab! Upload your swing video and ask me anything about your form. I'll analyze it and provide detailed biomechanics insights.",
      timestamp: new Date(),
    },
  ]);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [feedbackText, setFeedbackText] = useState<string | null>(null);
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

  const handleVideoUpload = (file: File) => {
    setVideoFile(file);
    setFeedbackText(null);
    setIsProcessingVideo(true);
    
    // TODO: Call API to process video and get feedback
    // For now, simulate processing
    setTimeout(() => {
      setIsProcessingVideo(false);
      // Mock feedback text - will be replaced with real API response
      setFeedbackText('Based on your swing, I detected early extension and excessive head movement, which contributes to inconsistency. Your hip rotation is good at 42°, but shoulder rotation could be improved. Your score is 7/10.');
    }, 3000);
  };

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
            feedbackText={feedbackText}
            isProcessing={isProcessingVideo}
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
