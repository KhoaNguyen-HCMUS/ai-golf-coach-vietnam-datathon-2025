"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Video } from "lucide-react"
import VideoRecorder from "./VideoRecorder"

interface VideoUploadAreaProps {
  videoFile: File | null
  onUpload: (file: File) => void
  onReset?: () => void
}

export default function VideoUploadArea({ videoFile, onUpload, onReset }: VideoUploadAreaProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [mode, setMode] = useState<"upload" | "record">("upload")

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith("video/")) {
      onUpload(file)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.currentTarget.files?.[0]
    if (file) {
      onUpload(file)
    }
  }

  const handleRecordComplete = (file: File) => {
    onUpload(file)
  }

  return (
    <div className="mb-6">
      {videoFile ? (
        <div className="card-base p-4 sm:p-6 bg-blue-50/50 border border-blue-200/50 rounded-2xl">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 flex-1">
              <div className="flex-shrink-0">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                  <svg className="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
              </div>
              <div className="flex-grow">
                <p className="font-medium text-gray-900">{videoFile.name}</p>
                <p className="text-sm text-gray-600">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button
              onClick={() => {
                onReset?.()
                setMode("upload")
              }}
              className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors"
            >
              Thay đổi
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Mode Toggle */}
          <div className="flex gap-2 rounded-lg border border-gray-200 bg-gray-50 p-1.5">
            <button
              onClick={() => setMode("upload")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-md font-semibold transition-all ${
                mode === "upload"
                  ? "bg-white text-blue-600 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <Upload className="h-4 w-4" />
              Upload Video
            </button>
            <button
              onClick={() => setMode("record")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-md font-semibold transition-all ${
                mode === "record"
                  ? "bg-white text-blue-600 shadow-sm"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <Video className="h-4 w-4" />
              Quay Video
            </button>
          </div>

          {/* Upload Mode */}
          {mode === "upload" && (
            <label
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`card-base block cursor-pointer p-8 text-center transition-all sm:p-12 border-2 rounded-2xl ${
                isDragging
                  ? "border-blue-400 bg-blue-50"
                  : "border-blue-200/50 bg-white hover:border-blue-300/50 hover:bg-blue-50/30"
              }`}
            >
              <div className="mx-auto mb-3 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100">
                <Upload className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="font-semibold text-gray-900">Upload your swing video</h3>
              <p className="mt-1 text-sm text-gray-600">Drag and drop or click to select (.mp4, .mov, .webm)</p>
              <input type="file" accept="video/*" onChange={handleFileSelect} className="hidden" />
            </label>
          )}

          {/* Record Mode */}
          {mode === "record" && <VideoRecorder onRecordComplete={handleRecordComplete} />}
        </div>
      )}
    </div>
  )
}
