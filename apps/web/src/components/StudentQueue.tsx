"use client"

import type { StudentVideo } from "../types"

interface StudentQueueProps {
  students: StudentVideo[]
  selectedStudent: StudentVideo | null
  onSelectStudent: (student: StudentVideo) => void
}

export default function StudentQueue({ students, selectedStudent, onSelectStudent }: StudentQueueProps) {
  const pendingCount = students.filter((s) => s.status === "pending").length

  return (
    <div className="card-base w-full lg:w-80 flex flex-col overflow-hidden rounded-none lg:rounded-lg m-0 lg:m-0 p-0 lg:p-0">
      <div className="border-b border-gray-200 bg-gradient-to-r from-cyan-50 to-blue-50 p-4">
        <h2 className="font-bold text-foreground text-lg">Review Queue</h2>
        <p className="mt-1 text-sm text-gray-600">{pendingCount} videos pending</p>
      </div>

      {/* Queue List */}
      <div className="scrollbar-hidden flex-1 overflow-y-auto divide-y divide-gray-100 max-h-96 lg:max-h-none">
        {students.map((student) => (
          <button
            key={student.id}
            onClick={() => onSelectStudent(student)}
            className={`w-full p-3 sm:p-4 text-left transition-all ${
              selectedStudent?.id === student.id ? "bg-cyan-50 border-l-4 border-cyan-500" : "hover:bg-gray-50"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <h3 className="truncate font-semibold text-foreground text-sm sm:text-base">{student.studentName}</h3>
                <p className="truncate text-xs sm:text-sm text-gray-600">{student.issue}</p>
                <p className="mt-1 text-xs text-gray-500">
                  {Math.round((Date.now() - student.timestamp.getTime()) / (60 * 1000))} min ago
                </p>
              </div>
              {student.status === "pending" && (
                <div className="mt-1 inline-flex h-3 w-3 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 flex-shrink-0 animate-pulse"></div>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
