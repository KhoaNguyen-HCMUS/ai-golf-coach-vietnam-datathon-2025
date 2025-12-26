"use client"

import { useState, useRef, useEffect } from "react"
import ChatInterface from "./ChatInterface"
import VideoUploadArea from "./VideoUploadArea"
import type { Message, AnalysisData } from "../types"

export default function PlayerMode() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Welcome to SwingAI Lab! Upload your swing video and ask me anything about your form. I'll analyze it and provide detailed biomechanics insights.",
      timestamp: new Date(),
    },
  ])
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleVideoUpload = (file: File) => {
    setVideoFile(file)
    const newMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: `Uploaded swing video: ${file.name}`,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
  }

  const handleSendMessage = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMessage])

    setIsAnalyzing(true)

    // Simulate API call - replace with real API
    setTimeout(() => {
      const mockAnalysis: AnalysisData = {
        keyframes: [
          { id: "p1", label: "P1", timestamp: 0 },
          { id: "p2", label: "P2", timestamp: 100 },
          { id: "p3", label: "P3", timestamp: 200 },
          { id: "p4", label: "P4", timestamp: 300 },
          { id: "p5", label: "P5", timestamp: 400 },
          { id: "p6", label: "P6", timestamp: 500 },
          { id: "p7", label: "P7", timestamp: 600 },
          { id: "p8", label: "P8", timestamp: 700 },
        ],
        metrics: {
          spinAngle: 2800,
          headMovement: 1.2,
          shoulderRotation: 87,
          hipRotation: 42,
        },
        diagnosis:
          "Based on your swing, I detected early extension and excessive head movement, which contributes to inconsistency. Your hip rotation is good at 42°, but shoulder rotation could be improved.",
      }

      setAnalysisData(mockAnalysis)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: mockAnalysis.diagnosis,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMessage])
      setIsAnalyzing(false)
    }, 2000)
  }

  return (
    <div className="flex min-h-[calc(100vh-80px)] flex-col gap-4 p-4 sm:gap-6 sm:p-6 bg-gradient-to-br from-white to-gray-50">
      <div className="mx-auto w-full max-w-4xl">
        <div className="mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
            The AI Lab
          </h1>
          <p className="mt-2 text-gray-600">Upload your swing and get instant AI-powered biomechanics analysis</p>
        </div>

        {/* Video Upload Section */}
        <VideoUploadArea videoFile={videoFile} onUpload={handleVideoUpload} />

        {/* Chat Interface */}
        <ChatInterface
          messages={messages}
          analysisData={analysisData}
          isAnalyzing={isAnalyzing}
          onSendMessage={handleSendMessage}
          messagesEndRef={messagesEndRef}
        />
      </div>
    </div>
  )
}
