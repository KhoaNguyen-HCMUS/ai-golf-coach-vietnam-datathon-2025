"use client"

import { useState } from "react"
import StudentQueue from "./StudentQueue"
import AnalysisDashboard from "./AnalysisDashboard"
import type { StudentVideo } from "../types"

export default function CoachMode() {
  const [selectedStudent, setSelectedStudent] = useState<StudentVideo | null>(null)
  const [students] = useState<StudentVideo[]>([
    {
      id: "1",
      studentName: "Quân Anh",
      issue: "Slice Error",
      status: "pending",
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      analysis: {
        keyframes: Array.from({ length: 8 }, (_, i) => ({
          id: `p${i + 1}`,
          label: `P${i + 1}`,
          timestamp: i * 100,
        })),
        metrics: {
          spinAngle: 2800,
          headMovement: 1.5,
          shoulderRotation: 82,
          hipRotation: 38,
        },
        diagnosis: "Early extension detected. Club face open at impact.",
      },
    },
    {
      id: "2",
      studentName: "Minh Tuấn",
      issue: "Hook Shot",
      status: "pending",
      timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000),
      analysis: {
        keyframes: Array.from({ length: 8 }, (_, i) => ({
          id: `p${i + 1}`,
          label: `P${i + 1}`,
          timestamp: i * 100,
        })),
        metrics: {
          spinAngle: 3200,
          headMovement: 0.8,
          shoulderRotation: 95,
          hipRotation: 52,
        },
        diagnosis: "Over-rotation in downswing. Need better hip-shoulder separation.",
      },
    },
    {
      id: "3",
      studentName: "Linh Đan",
      issue: "Inconsistent Distance",
      status: "pending",
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
      analysis: {
        keyframes: Array.from({ length: 8 }, (_, i) => ({
          id: `p${i + 1}`,
          label: `P${i + 1}`,
          timestamp: i * 100,
        })),
        metrics: {
          spinAngle: 2500,
          headMovement: 2.1,
          shoulderRotation: 78,
          hipRotation: 35,
        },
        diagnosis: "Excessive head movement causing poor ball contact.",
      },
    },
  ])

  return (
    <div className="flex min-h-[calc(100vh-80px)] flex-col lg:flex-row gap-0 lg:gap-4 lg:p-6 bg-gray-100">
      {/* Sidebar - Student Queue */}
      <StudentQueue students={students} selectedStudent={selectedStudent} onSelectStudent={setSelectedStudent} />

      {/* Main Content - Analysis Dashboard */}
      <div className="flex-1 min-h-0">
        {selectedStudent ? (
          <AnalysisDashboard student={selectedStudent} />
        ) : (
          <div className="card-base flex h-full items-center justify-center m-0 lg:m-0">
            <div className="text-center">
              <div className="mx-auto mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-cyan-100">
                <svg className="h-8 w-8 text-cyan-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-foreground">Select a student</h3>
              <p className="mt-1 text-gray-600">Choose a video from the queue to review</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
