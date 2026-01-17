'use client';

import type React from 'react';

import { useState } from 'react';
import type { Message } from '../types';
import { Send, Trash2, X } from 'lucide-react';

interface ChatInterfaceProps {
  messages: Message[];
  isAnalyzing: boolean;
  onSendMessage: (message: string) => void;
  onClearMessages?: () => void;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  analysisContext?: string | null;
  onClearContext?: () => void;
}

export default function ChatInterface({
  messages,
  isAnalyzing,
  onSendMessage,
  onClearMessages,
  messagesEndRef,
  analysisContext,
  onClearContext,
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim()) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  return (
    <div className='flex h-[600px] flex-col overflow-hidden rounded-2xl border border-gray-300/50 bg-gradient-to-b from-white to-gray-50 shadow-lg'>
      {/* Header with Clear button */}
      <div className='flex items-center justify-between border-b border-gray-200/50 bg-gradient-to-r from-blue-50 to-cyan-50 px-4 py-3 sm:px-6 sm:py-4'>
        <h3 className='text-sm font-semibold text-gray-700 uppercase tracking-wide'>AI Coach Chat</h3>
        {onClearMessages && messages.length > 1 && (
          <button
            onClick={onClearMessages}
            className='flex items-center gap-2 px-3 py-1.5 text-xs text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors'
            title='Clear chat history'
          >
            <Trash2 className='h-4 w-4' />
            Clear
          </button>
        )}
      </div>

      <div className='scrollbar-hidden flex-1 overflow-y-auto p-4 sm:p-6'>
        <div className='space-y-4'>
          {/* Analysis Context Label */}
          {analysisContext && (
            <div className='mb-4 p-4 bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-xl'>
              <div className='flex items-start justify-between gap-3'>
                <div className='flex-1'>
                  <div className='flex items-center gap-2 mb-2'>
                    <span className='text-xs font-semibold text-blue-700 uppercase tracking-wide'>
                      Analysis Context
                    </span>
                  </div>
                  <div
                    className='text-sm text-gray-700 prose prose-sm max-w-none'
                    dangerouslySetInnerHTML={{ __html: analysisContext }}
                    style={{
                      fontSize: '0.875rem',
                      lineHeight: '1.6',
                    }}
                  />
                </div>
                {onClearContext && (
                  <button
                    onClick={onClearContext}
                    className='flex-shrink-0 p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors'
                    title='Clear analysis context'
                  >
                    <X className='h-4 w-4' />
                  </button>
                )}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-xs rounded-xl px-4 py-3 sm:max-w-md transition-all ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-600 text-white shadow-md shadow-blue-200/50'
                    : 'bg-gray-100 border border-gray-200 text-gray-900'
                }`}
              >
                {message.role === 'assistant' && message.content.includes('<') ? (
                  <div
                    className='text-sm leading-relaxed prose prose-sm max-w-none'
                    dangerouslySetInnerHTML={{ __html: message.content }}
                  />
                ) : (
                  <p className='text-sm leading-relaxed'>{message.content}</p>
                )}
              </div>
            </div>
          ))}

          {isAnalyzing && (
            <div className='flex justify-start'>
              <div className='rounded-xl bg-gray-100 border border-gray-200 px-4 py-3'>
                <div className='flex gap-2'>
                  <div className='h-2 w-2 animate-bounce rounded-full bg-blue-500'></div>
                  <div className='animation-delay-200 h-2 w-2 animate-bounce rounded-full bg-blue-500'></div>
                  <div className='animation-delay-400 h-2 w-2 animate-bounce rounded-full bg-blue-500'></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        className='border-t border-gray-200/50 bg-gradient-to-t from-gray-50 to-white p-4 sm:p-6'
      >
        <div className='flex gap-3'>
          <input
            type='text'
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder='Ask your AI coach a question...'
            className='flex-1 rounded-xl border border-gray-300/50 bg-white px-4 py-3 text-gray-900 placeholder-gray-500 focus:border-blue-400/50 focus:outline-none focus:ring-2 focus:ring-blue-400/20 transition-all'
            disabled={isAnalyzing}
          />
          <button
            type='submit'
            disabled={isAnalyzing || !inputValue.trim()}
            className='rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-5 py-3 font-semibold text-white shadow-md shadow-blue-200/40 hover:shadow-blue-300/60 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-95'
            title='Send message'
          >
            <Send className='h-5 w-5' />
          </button>
        </div>
      </form>
    </div>
  );
}
