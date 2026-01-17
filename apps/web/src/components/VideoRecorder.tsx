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
      setError("Trình duyệt của bạn không hỗ trợ truy cập camera. Vui lòng sử dụng Chrome, Firefox, Safari hoặc Edge.")
      setHasPermission(false)
      setIsLoading(false)
      return
    }

    if (!window.MediaRecorder) {
      setError("Trình duyệt của bạn không hỗ trợ quay video. Vui lòng cập nhật trình duyệt.")
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
        let errorMessage = "Không thể truy cập camera."
        
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          errorMessage = "Quyền truy cập camera bị từ chối. Vui lòng cấp quyền trong cài đặt trình duyệt."
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          errorMessage = "Không tìm thấy camera. Vui lòng kiểm tra kết nối camera."
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
          errorMessage = "Camera đang được sử dụng bởi ứng dụng khác. Vui lòng đóng ứng dụng đó."
        } else if (err.message === 'SECURE_CONTEXT_REQUIRED') {
          errorMessage = "Camera chỉ hoạt động trên HTTPS hoặc localhost. Vui lòng truy cập qua localhost hoặc HTTPS."
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = "Camera không hỗ trợ độ phân giải yêu cầu. Đang thử cài đặt mặc định..."
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
            errorMessage = "Không thể truy cập camera với bất kỳ cài đặt nào."
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
      setError("Camera chưa sẵn sàng")
      return
    }

    try {
      chunksRef.current = []
      const mimeType = getSupportedMimeType()
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {}
      
      const mediaRecorder = new MediaRecorder(streamRef.current, options)
      
      // Check if MediaRecorder is actually supported
      if (!mediaRecorder) {
        throw new Error('MediaRecorder không được hỗ trợ')
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
        setError('Lỗi khi quay video. Vui lòng thử lại.')
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
      setError("Không thể bắt đầu quay video")
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
        <h3 className="font-semibold text-gray-900 mb-2">Đang tải camera...</h3>
        <p className="text-sm text-gray-600">Vui lòng cho phép truy cập camera khi được hỏi</p>
      </div>
    )
  }

  if (hasPermission === false) {
    return (
      <div className="card-base p-8 text-center border border-red-200 bg-red-50/50 rounded-2xl">
        <div className="mx-auto mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-red-100">
          <Video className="h-6 w-6 text-red-600" />
        </div>
        <h3 className="font-semibold text-gray-900 mb-2">Không thể truy cập camera</h3>
        <p className="text-sm text-gray-600 mb-4 whitespace-pre-line">{error || "Vui lòng cấp quyền truy cập camera trong trình duyệt"}</p>
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
                  setError("Quyền bị từ chối. Vui lòng:\n1. Click vào icon khóa ở thanh địa chỉ\n2. Cho phép camera\n3. Refresh trang")
                } else {
                  setError(err.message || "Không thể truy cập camera")
                }
                setHasPermission(false)
              } finally {
                setIsLoading(false)
              }
            }}
            className="rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-6 py-2.5 font-semibold text-white shadow-md hover:shadow-lg transition-all"
          >
            Thử lại
          </button>
          <button
            onClick={() => window.location.reload()}
            className="block w-full mt-2 text-sm text-gray-600 hover:text-gray-900 underline"
          >
            Tải lại trang
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
                  Bắt đầu quay
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-red-600 px-6 py-3 font-semibold text-white shadow-md hover:shadow-lg transition-all"
                >
                  <Square className="h-5 w-5" />
                  Dừng quay
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
                Sử dụng video này
              </button>
              <button
                onClick={resetRecording}
                className="flex items-center justify-center gap-2 rounded-xl border border-gray-300 bg-white px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 transition-all"
              >
                <RotateCcw className="h-5 w-5" />
                Quay lại
              </button>
              <a
                href={recordedUrl}
                download={`swing_${Date.now()}.${recordedBlob?.type.includes('mp4') ? 'mp4' : 'webm'}`}
                className="flex items-center justify-center gap-2 rounded-xl border border-gray-300 bg-white px-6 py-3 font-semibold text-gray-700 hover:bg-gray-50 transition-all"
              >
                <Download className="h-5 w-5" />
                Tải xuống
              </a>
            </>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-4 p-3 bg-blue-50/50 rounded-lg border border-blue-100">
          <p className="text-xs text-gray-600">
            {!recordedUrl
              ? "💡 Đặt camera để có thể nhìn thấy toàn bộ cú swing của bạn. Nhấn 'Bắt đầu quay' khi sẵn sàng."
              : "✅ Video đã được ghi lại. Bạn có thể xem lại, tải xuống hoặc sử dụng để phân tích."}
          </p>
        </div>
      </div>
    </div>
  )
}

