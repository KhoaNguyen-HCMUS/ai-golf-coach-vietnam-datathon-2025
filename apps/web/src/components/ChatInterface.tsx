"use client"

import type React from "react"

import { useState } from "react"
import type { Message, AnalysisData } from "../types"
import AnalysisWidget from "./AnalysisWidget"
import { Send } from "lucide-react"

interface ChatInterfaceProps {
  messages: Message[]
  analysisData: AnalysisData | null
  isAnalyzing: boolean
  onSendMessage: (message: string) => void
  messagesEndRef: React.RefObject<HTMLDivElement>
}

export default function ChatInterface({
  messages,
  analysisData,
  isAnalyzing,
  onSendMessage,
  messagesEndRef,
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (inputValue.trim()) {
      onSendMessage(inputValue)
      setInputValue("")
    }
  }

  return (
    <div className="flex h-[600px] flex-col overflow-hidden rounded-2xl border border-gray-300/50 bg-gradient-to-b from-white to-gray-50 shadow-lg">
      <div className="scrollbar-hidden flex-1 overflow-y-auto p-4 sm:p-6">
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-xs rounded-xl px-4 py-3 sm:max-w-md transition-all ${
                  message.role === "user"
                    ? "bg-gradient-to-r from-blue-500 to-cyan-600 text-white shadow-md shadow-blue-200/50"
                    : "bg-gray-100 border border-gray-200 text-gray-900"
                }`}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
              </div>
            </div>
          ))}

          {isAnalyzing && (
            <div className="flex justify-start">
              <div className="rounded-xl bg-gray-100 border border-gray-200 px-4 py-3">
                <div className="flex gap-2">
                  <div className="h-2 w-2 animate-bounce rounded-full bg-blue-500"></div>
                  <div className="animation-delay-200 h-2 w-2 animate-bounce rounded-full bg-blue-500"></div>
                  <div className="animation-delay-400 h-2 w-2 animate-bounce rounded-full bg-blue-500"></div>
                </div>
              </div>
            </div>
          )}

          {analysisData && (
            <div className="mt-4">
              <AnalysisWidget data={analysisData} />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-gray-200/50 bg-gradient-to-t from-gray-50 to-white p-4 sm:p-6"
      >
        <div className="flex gap-3">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask your AI coach a question..."
            className="flex-1 rounded-xl border border-gray-300/50 bg-white px-4 py-3 text-gray-900 placeholder-gray-500 focus:border-blue-400/50 focus:outline-none focus:ring-2 focus:ring-blue-400/20 transition-all"
            disabled={isAnalyzing}
          />
          <button
            type="submit"
            disabled={isAnalyzing || !inputValue.trim()}
            className="rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-5 py-3 font-semibold text-white shadow-md shadow-blue-200/40 hover:shadow-blue-300/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95"
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
      </form>
    </div>
  )
}
