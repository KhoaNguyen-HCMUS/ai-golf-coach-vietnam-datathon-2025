import type { AnalysisData } from "../types"
import { Send } from "lucide-react"

interface AnalysisWidgetProps {
  data: AnalysisData
}

export default function AnalysisWidget({ data }: AnalysisWidgetProps) {
  return (
    <div className="space-y-4">
      <div className="rounded-xl bg-gradient-to-br from-blue-50/50 to-cyan-50/50 border border-blue-200/50 p-5">
        <h4 className="mb-4 font-semibold text-gray-900 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-blue-500"></span>
          Swing Sequence (8 Key Frames)
        </h4>
        <div className="grid grid-cols-4 gap-2 sm:gap-3 md:grid-cols-8">
          {data.keyframes.map((frame) => (
            <div
              key={frame.id}
              className="group aspect-square cursor-pointer overflow-hidden rounded-lg border border-blue-200/50 bg-white transition-all hover:border-blue-400/50 hover:shadow-md hover:shadow-blue-200/50"
            >
              <div className="flex h-full items-center justify-center bg-gradient-to-br from-blue-100/30 to-cyan-100/30 group-hover:from-blue-100/50 group-hover:to-cyan-100/50 transition-all">
                <span className="text-xs font-bold text-blue-600 sm:text-sm group-hover:scale-110 transition-transform">
                  {frame.label}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl bg-gradient-to-br from-blue-50/50 to-cyan-50/50 border border-blue-200/50 p-5">
        <h4 className="mb-4 font-semibold text-gray-900 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-500"></span>
          Biomechanics Metrics
        </h4>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="flex flex-col rounded-lg bg-white p-3 border border-blue-200/50">
            <span className="text-xs text-gray-600 font-medium">Spin Angle</span>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent sm:text-2xl mt-1">
              {data.metrics.spinAngle}
            </span>
            <span className="text-xs text-gray-500">RPM</span>
          </div>
          <div className="flex flex-col rounded-lg bg-white p-3 border border-blue-200/50">
            <span className="text-xs text-gray-600 font-medium">Head Movement</span>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent sm:text-2xl mt-1">
              {data.metrics.headMovement.toFixed(1)}
            </span>
            <span className="text-xs text-gray-500">inches</span>
          </div>
          <div className="flex flex-col rounded-lg bg-white p-3 border border-blue-200/50">
            <span className="text-xs text-gray-600 font-medium">Shoulder Rot.</span>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent sm:text-2xl mt-1">
              {data.metrics.shoulderRotation}°
            </span>
          </div>
          <div className="flex flex-col rounded-lg bg-white p-3 border border-blue-200/50">
            <span className="text-xs text-gray-600 font-medium">Hip Rotation</span>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent sm:text-2xl mt-1">
              {data.metrics.hipRotation}°
            </span>
          </div>
        </div>
      </div>

      <button className="w-full rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-6 py-3.5 font-semibold text-white shadow-md shadow-blue-200/40 hover:shadow-blue-300/60 transition-all hover:scale-105 active:scale-95 flex items-center justify-center gap-2">
        <Send className="h-5 w-5" />
        Send to My Coach
      </button>
    </div>
  )
}
