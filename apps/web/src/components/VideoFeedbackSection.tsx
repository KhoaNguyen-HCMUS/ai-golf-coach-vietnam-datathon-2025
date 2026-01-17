"use client"

import type React from "react"
import { Video, Loader2, Download } from "lucide-react"
import { useEffect } from "react"

interface VideoFeedbackSectionProps {
  videoFileName: string | null
  feedbackHTML: string | null
  isProcessing: boolean
  isConnected?: boolean
  videoData?: {
    base64: string;
    mimeType: string;
    filename: string;
    size: number;
  }[];
}

export default function VideoFeedbackSection({
  videoFileName,
  feedbackHTML,
  isProcessing,
  isConnected = true,
  videoData = [],
}: VideoFeedbackSectionProps) {
  // Debug logging
  useEffect(() => {
    console.log('🟠 [VideoFeedbackSection] Props changed:', {
      videoFileName,
      hasFeedbackHTML: !!feedbackHTML,
      feedbackHTMLLength: feedbackHTML?.length || 0,
      isProcessing,
      isConnected,
      videoDataCount: videoData.length
    });
  }, [videoFileName, feedbackHTML, isProcessing, isConnected, videoData]);

  // Download video function
  const downloadVideo = (base64: string, filename: string, mimeType: string) => {
    try {
      // Convert base64 to blob
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: mimeType });
      
      // Create download link
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      console.log(`✓ Downloaded video: ${filename}`);
    } catch (error) {
      console.error('✗ Failed to download video:', error);
    }
  };
  return (
    <div className="rounded-2xl border border-gray-300/50 bg-gradient-to-b from-white to-gray-50 shadow-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-200/50 bg-gradient-to-r from-blue-50 to-cyan-50 px-4 py-3 sm:px-6 sm:py-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Video Analysis</h3>
          {videoFileName && (
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-xs text-gray-600">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          )}
        </div>
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
            {feedbackHTML && !isProcessing && (
              <div className="space-y-4">
                <div className="p-4 sm:p-6 bg-gray-50 border border-gray-200 rounded-xl">
                  <div
                    className="analysis-html"
                    dangerouslySetInnerHTML={{ __html: feedbackHTML }}
                    style={{
                      color: '#1f2937',
                      fontSize: '0.875rem',
                      lineHeight: '1.6',
                    }}
                  />
                </div>

                {/* Video Players and Download Buttons */}
                {videoData.length > 0 && (
                  <div className="space-y-4">
                    {videoData.map((video, index) => (
                      <div key={index} className="p-4 bg-white border border-gray-200 rounded-xl">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="text-sm font-semibold text-gray-900">
                            Hit #{index + 1} Video
                          </h4>
                          <button
                            onClick={() => downloadVideo(video.base64, video.filename, video.mimeType)}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                          >
                            <Download className="h-4 w-4" />
                            Download
                          </button>
                        </div>
                        <video
                          controls
                          className="w-full rounded-lg"
                          style={{ maxHeight: '400px' }}
                        >
                          <source
                            src={`data:${video.mimeType};base64,${video.base64}`}
                            type={video.mimeType}
                          />
                          Your browser does not support the video tag.
                        </video>
                        <p className="text-xs text-gray-500 mt-2">
                          Size: {(video.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Empty State - Waiting for feedback */}
            {!feedbackHTML && !isProcessing && (
              <div className="text-center py-8">
                <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-gray-100">
                  <Video className="h-6 w-6 text-gray-400" />
                </div>
                <p className="text-sm text-gray-500">
                  {isConnected ? 'Waiting for analysis feedback...' : 'WebSocket disconnected. Please refresh the page.'}
                </p>
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

