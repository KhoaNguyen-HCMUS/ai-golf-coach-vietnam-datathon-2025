"use client"

import { useState } from "react"
import type { StudentVideo } from "../types"

interface AnalysisDashboardProps {
  student: StudentVideo
}

export default function AnalysisDashboard({ student }: AnalysisDashboardProps) {
  const [coachFeedback, setCoachFeedback] = useState("")
  const [selectedFrames, setSelectedFrames] = useState<string[]>([])
  const [isEditing, setIsEditing] = useState(false)
  const [isReviewed, setIsReviewed] = useState(false)

  const handleFrameSelect = (frameId: string) => {
    setSelectedFrames((prev) => (prev.includes(frameId) ? prev.filter((id) => id !== frameId) : [...prev, frameId]))
  }

  return (
    <div className="flex h-full flex-col gap-4 p-4 sm:p-6 lg:p-0">
      <div className="card-base p-4 sm:p-6 bg-gradient-to-r from-white to-cyan-50">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold text-foreground">{student.studentName}</h2>
            <p className="mt-1 text-sm sm:text-base text-gray-600">Issue: {student.issue}</p>
          </div>
          <div className="text-left sm:text-right">
            <span
              className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${
                isReviewed ? "bg-green-100 text-green-700" : "bg-cyan-100 text-cyan-700"
              }`}
            >
              {isReviewed ? "✓ Reviewed" : "⏳ Pending"}
            </span>
          </div>
        </div>
      </div>

      <div className="flex flex-1 gap-4 min-h-0 flex-col lg:flex-row overflow-hidden">
        {/* Left: Video Player & Keyframes */}
        <div className="card-base flex-1 flex flex-col overflow-hidden rounded-lg p-0">
          <div className="flex-1 bg-gradient-to-br from-gray-50 to-gray-100 relative overflow-hidden min-h-48 sm:min-h-64">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center px-4">
                <svg
                  className="mx-auto h-12 w-12 sm:h-16 sm:w-16 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
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
                <p className="mt-2 text-gray-600 font-medium text-sm sm:text-base">Video Player</p>
              </div>
            </div>
          </div>

          {/* Keyframes Timeline */}
          <div className="border-t border-gray-200 p-4 bg-white">
            <p className="mb-3 text-xs font-bold text-gray-700 uppercase tracking-wide">Keyframes</p>
            <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2">
              {student.analysis?.keyframes.map((frame) => (
                <button
                  key={frame.id}
                  onClick={() => isEditing && handleFrameSelect(frame.id)}
                  className={`aspect-square rounded-lg border-2 transition-all font-bold text-xs sm:text-sm ${
                    selectedFrames.includes(frame.id)
                      ? "border-cyan-500 bg-cyan-50 text-cyan-700"
                      : "border-gray-300 bg-white hover:border-cyan-400 text-gray-600"
                  } ${isEditing ? "cursor-pointer" : "cursor-default"}`}
                >
                  {frame.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: AI Diagnosis & Coach Validation */}
        <div className="flex w-full flex-col gap-4 lg:w-96 lg:overflow-y-auto">
          {/* AI Diagnosis */}
          <div className="card-base flex flex-col p-4 sm:p-6">
            <h3 className="mb-4 font-bold text-foreground text-base sm:text-lg flex items-center gap-2">
              <span className="text-lg sm:text-xl">🤖</span> AI Diagnosis
            </h3>
            <p className="mb-4 text-xs sm:text-sm text-foreground leading-relaxed">{student.analysis?.diagnosis}</p>

            {/* Metrics */}
            <div className="space-y-3 rounded-lg bg-gradient-to-br from-cyan-50 to-blue-50 p-3 sm:p-4 border border-cyan-100">
              <div className="flex items-center justify-between">
                <span className="text-xs sm:text-sm font-medium text-gray-700">Spin Angle</span>
                <span className="font-bold text-cyan-600 text-sm sm:text-lg">
                  {student.analysis?.metrics.spinAngle} RPM
                </span>
              </div>
              <div className="h-px bg-gray-200"></div>
              <div className="flex items-center justify-between">
                <span className="text-xs sm:text-sm font-medium text-gray-700">Head Movement</span>
                <span className="font-bold text-cyan-600 text-sm sm:text-lg">
                  {student.analysis?.metrics.headMovement.toFixed(1)}"
                </span>
              </div>
              <div className="h-px bg-gray-200"></div>
              <div className="flex items-center justify-between">
                <span className="text-xs sm:text-sm font-medium text-gray-700">Shoulder Rotation</span>
                <span className="font-bold text-cyan-600 text-sm sm:text-lg">
                  {student.analysis?.metrics.shoulderRotation}°
                </span>
              </div>
              <div className="h-px bg-gray-200"></div>
              <div className="flex items-center justify-between">
                <span className="text-xs sm:text-sm font-medium text-gray-700">Hip Rotation</span>
                <span className="font-bold text-cyan-600 text-sm sm:text-lg">
                  {student.analysis?.metrics.hipRotation}°
                </span>
              </div>
            </div>
          </div>

          {/* Coach Validation */}
          <div className="card-base flex flex-col p-4 sm:p-6">
            <h3 className="mb-4 font-bold text-foreground text-base sm:text-lg flex items-center gap-2">
              <span className="text-lg sm:text-xl">👨‍🏫</span> Coach Validation
            </h3>

            {isEditing ? (
              <>
                <textarea
                  value={coachFeedback}
                  onChange={(e) => setCoachFeedback(e.target.value)}
                  placeholder="Enter your feedback or corrections..."
                  className="input-base mb-3 flex-1 resize-none text-xs sm:text-sm"
                  rows={4}
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setIsEditing(false)
                      setIsReviewed(true)
                    }}
                    className="btn-primary flex-1 text-sm sm:text-base"
                  >
                    Submit Feedback
                  </button>
                  <button onClick={() => setIsEditing(false)} className="btn-secondary flex-1 text-sm sm:text-base">
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <div className="space-y-2">
                <button
                  onClick={() => {
                    setIsEditing(false)
                    setIsReviewed(true)
                  }}
                  disabled={isReviewed}
                  className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed text-sm sm:text-base"
                >
                  ✓ Confirm AI Analysis
                </button>
                <button onClick={() => setIsEditing(true)} className="btn-secondary w-full text-sm sm:text-base">
                  ✏️ Edit & Override
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
