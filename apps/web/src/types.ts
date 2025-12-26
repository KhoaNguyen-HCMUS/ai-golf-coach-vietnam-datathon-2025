export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export interface Keyframe {
  id: string
  label: string
  timestamp: number
  angle?: number
  value?: string
}

export interface AnalysisData {
  keyframes: Keyframe[]
  metrics: {
    spinAngle: number
    headMovement: number
    shoulderRotation: number
    hipRotation: number
  }
  diagnosis: string
}

export interface StudentVideo {
  id: string
  studentName: string
  issue: string
  status: "pending" | "reviewed"
  timestamp: Date
  analysis?: AnalysisData
}
