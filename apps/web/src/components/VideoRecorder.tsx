"use client"

import { useState, useRef, useEffect } from "react"
import { Video, Square, RotateCcw, Download, Upload } from "lucide-react"

interface VideoRecorderProps {
  onRecordComplete: (file: File) => void
}

export default function VideoRecorder({ onRecordComplete }: VideoRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  // Check browser compatibility
  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Your browser does not support camera access. Please use Chrome, Firefox, Safari, or Edge.")
      setHasPermission(false)
      setIsLoading(false)
      return
    }

    if (!window.MediaRecorder) {
      setError("Your browser does not support video recording. Please update your browser.")
      setHasPermission(false)
      setIsLoading(false)
      return
    }
  }, [])

  // Request camera permission and start preview
  useEffect(() => {
    const startPreview = async () => {
      setIsLoading(true)
      try {
        // Check if we're on HTTPS or localhost
        const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1'
        
        if (!isSecureContext) {
          throw new Error('SECURE_CONTEXT_REQUIRED')
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user", // front camera
          },
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          // Wait for video to be ready
          await new Promise((resolve) => {
            if (videoRef.current) {
              videoRef.current.onloadedmetadata = () => resolve(undefined)
            } else {
              resolve(undefined)
            }
          })
        }
        setHasPermission(true)
        setError(null)
      } catch (err: any) {
        console.error("Error accessing camera:", err)
        let errorMessage = "Unable to access camera."
        
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          errorMessage = "Camera permission denied. Please grant permission in browser settings."
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          errorMessage = "Camera not found. Please check camera connection."
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          errorMessage = "Camera is being used by another application. Please close that application."
        } else if (err.message === 'SECURE_CONTEXT_REQUIRED') {
          errorMessage = "Camera only works on HTTPS or localhost. Please access via localhost or HTTPS."
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = "Camera does not support requested resolution. Trying default settings..."
          // Retry with default settings
          try {
            const stream = await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false,
            })
            streamRef.current = stream
            if (videoRef.current) {
              videoRef.current.srcObject = stream
            }
            setHasPermission(true)
            setError(null)
            setIsLoading(false)
            return
          } catch (retryErr) {
            errorMessage = "Unable to access camera with any settings."
          }
        }
        
        setError(errorMessage)
        setHasPermission(false)
      } finally {
        setIsLoading(false)
      }
    }

    startPreview()

    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  // Cleanup recorded URL when component unmounts or when reset
  useEffect(() => {
    return () => {
      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl)
      }
    }
  }, [recordedUrl])

  const getSupportedMimeType = () => {
    const types = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4',
    ]
    
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type
      }
    }
    return '' // Browser will choose default
  }

  const startRecording = () => {
    if (!streamRef.current) {
      setError("Camera is not ready")
      return
    }

    try {
      chunksRef.current = []
      const mimeType = getSupportedMimeType()
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {}
      
      const mediaRecorder = new MediaRecorder(streamRef.current, options)
      
      // Check if MediaRecorder is actually supported
      if (!mediaRecorder) {
        throw new Error('MediaRecorder is not supported')
      }

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType || 'video/webm'
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const url = URL.createObjectURL(blob)
        setRecordedBlob(blob)
        setRecordedUrl(url)
        setIsRecording(false)
        setRecordingTime(0)
        if (timerRef.current) {
          clearInterval(timerRef.current)
        }
      }

      mediaRecorder.onerror = (event: any) => {
        console.error('MediaRecorder error:', event)
        setError('Error recording video. Please try again.')
        setIsRecording(false)
        if (timerRef.current) {
          clearInterval(timerRef.current)
        }
      }

      mediaRecorderRef.current = mediaRecorder
      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)
      setRecordingTime(0)
      setError(null)

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } catch (err) {
      console.error("Error starting recording:", err)
      setError("Unable to start recording")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  const resetRecording = () => {
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl)
    }
    setRecordedBlob(null)
    setRecordedUrl(null)
    setRecordingTime(0)
    setError(null)
    // Restart preview if stream is still available
    if (streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }

  const handleUseVideo = () => {
    if (recordedBlob) {
      // Get the actual mime type from the blob
      const mimeType = recordedBlob.type || 'video/webm'
      const extension = mimeType.includes('mp4') ? 'mp4' : 'webm'
      // Convert blob to File
      const file = new File([recordedBlob], `swing_${Date.now()}.${extension}`, {
        type: mimeType,
      })
      onRecordComplete(file)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  if (isLoading) {
    return (
      <div className="card-base p-8 text-center border border-blue-200 bg-blue-50/50 rounded-2xl">
        <div className="mx-auto mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 animate-pulse">
          <Video className="h-6 w-6 text-blue-600" />
        </div>
        <h3 className="font-semibold text-gray-900 mb-2">Loading camera...</h3>
        <p className="text-sm text-gray-600">Please allow camera access when prompted</p>
      </div>
    )
  }

  if (hasPermission === false) {
    return (
      <div className="card-base p-8 text-center border border-red-200 bg-red-50/50 rounded-2xl">
        <div className="mx-auto mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-red-100">
          <Video className="h-6 w-6 text-red-600" />
        </div>
        <h3 className="font-semibold text-gray-900 mb-2">Unable to access camera</h3>
        <p className="text-sm text-gray-600 mb-4 whitespace-pre-line">{error || "Please grant camera permission in browser settings"}</p>
        <div className="space-y-2">
          <button
            onClick={async () => {
              setIsLoading(true)
              setError(null)
              try {
                const stream = await navigator.mediaDevices.getUserMedia({
                  video: true,
                  audio: false,
                })
                streamRef.current = stream
                if (videoRef.current) {
                  videoRef.current.srcObject = stream
                }
                setHasPermission(true)
                setError(null)
              } catch (err: any) {
                if (err.name === 'NotAllowedError') {
                  setError("Permission denied. Please:\n1. Click the lock icon in the address bar\n2. Allow camera access\n3. Refresh the page")
                } else {
                  setError(err.message || "Unable to access camera")
                }
                setHasPermission(false)
              } finally {
                setIsLoading(false)
              }
            }}
            className="rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-6 py-2.5 font-semibold text-white shadow-md hover:shadow-lg transition-all"
          >
            Try Again
          </button>
          <button
            onClick={() => window.location.reload()}
            className="block w-full mt-2 text-sm text-gray-600 hover:text-gray-900 underline"
          >
            Reload Page
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="mb-6">
      <div className="card-base p-4 sm:p-6 border border-blue-200/50 rounded-2xl bg-gradient-to-br from-white to-blue-50/30">
        {/* Video Preview/Recording */}
        <div className="relative mb-4 rounded-xl overflow-hidden bg-black aspect-video">
          {recordedUrl ? (
            <video src={recordedUrl} controls className="w-full h-full object-contain" />
          ) : (
            <>
              <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
              {isRecording && (
                <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1.5 rounded-full">
                  <div className="h-2 w-2 bg-white rounded-full animate-pulse"></div>
                  <span className="text-sm font-semibold">{formatTime(recordingTime)}</span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3">
          {!recordedUrl ? (
            <>
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={!hasPermission}
                  className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-red-600 px-6 py-3 font-semibold text-white shadow-md hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Video className="h-5 w-5" />
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-red-600 px-6 py-3 font-semibold text-white shadow-md hover:shadow-lg transition-all"
                >
                  <Square className="h-5 w-5" />
                  Stop Recording
                </button>
              )}
            </>
          ) : (
            <>
              <button
                onClick={handleUseVideo}
                className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-6 py-3 font-semibold text-white shadow-md hover:shadow-lg transition-all"
              >
                <Upload className="h-5 w-5" />
                Use This Video
              </button>
              <button
                onClick={resetRecording}
                className="flex items-center justify-center gap-2 rounded-xl border border-gray-300 bg-white px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 transition-all"
              >
                <RotateCcw className="h-5 w-5" />
                Record Again
              </button>
              <a
                href={recordedUrl}
                download={`swing_${Date.now()}.${recordedBlob?.type.includes('mp4') ? 'mp4' : 'webm'}`}
                className="flex items-center justify-center gap-2 rounded-xl border border-gray-300 bg-white px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 transition-all"
              >
                <Download className="h-5 w-5" />
                Download
              </a>
            </>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-4 p-3 bg-blue-50/50 rounded-lg border border-blue-100">
          <p className="text-xs text-gray-600">
            {!recordedUrl
              ? "Position the camera to see your entire swing. Press 'Start Recording' when ready."
              : "Video recorded successfully. You can review, download, or use it for analysis."}
          </p>
        </div>
      </div>
    </div>
  )
}

