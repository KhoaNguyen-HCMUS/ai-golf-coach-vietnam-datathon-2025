"use client"

import type React from "react"
import { Video, Loader2 } from "lucide-react"

interface VideoFeedbackSectionProps {
  videoFileName: string | null
  feedbackText: string | null
  isProcessing: boolean
}

export default function VideoFeedbackSection({
  videoFileName,
  feedbackText,
  isProcessing,
}: VideoFeedbackSectionProps) {
  return (
    <div className="rounded-2xl border border-gray-300/50 bg-gradient-to-b from-white to-gray-50 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200/50 bg-gradient-to-r from-blue-50 to-cyan-50 px-4 py-3 sm:px-6 sm:py-4">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Video Analysis</h3>
      </div>

      {/* Content */}
      <div className="p-4 sm:p-6">
        {videoFileName ? (
          <div className="space-y-4">
            {/* Video Upload Info */}
            <div className="flex items-start gap-3 p-4 bg-blue-50/50 border border-blue-200/50 rounded-xl">
              <div className="flex-shrink-0 mt-0.5">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100">
                  <Video className="h-5 w-5 text-blue-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">Uploaded swing video</p>
                <p className="text-sm text-gray-600 mt-1 break-all">{videoFileName}</p>
              </div>
            </div>

            {/* Processing State */}
            {isProcessing && (
              <div className="flex items-center justify-center gap-3 p-6 bg-cyan-50/50 border border-cyan-200/50 rounded-xl">
                <Loader2 className="h-5 w-5 text-cyan-600 animate-spin" />
                <p className="text-sm text-gray-700">Processing...</p>
              </div>
            )}

            {/* Feedback from Server */}
            {feedbackText && !isProcessing && (
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-xl">
                <p className="text-sm text-gray-900 whitespace-pre-wrap leading-relaxed">{feedbackText}</p>
              </div>
            )}

            {/* Empty State - Waiting for feedback */}
            {!feedbackText && !isProcessing && (
              <div className="text-center py-8">
                <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gray-100">
                  <Video className="h-6 w-6 text-gray-400" />
                </div>
                <p className="text-sm text-gray-500">Waiting for analysis feedback...</p>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gray-100">
              <Video className="h-6 w-6 text-gray-400" />
            </div>
            <p className="text-sm text-gray-500">No video uploaded yet</p>
          </div>
        )}
      </div>
    </div>
  )
}

