"use client"

import { Home } from "lucide-react"
import Link from "next/link"

interface ModeSwitchProps {
  mode: "player" | "coach"
  setMode: (mode: "player" | "coach") => void
}

export default function ModeSwitch({ mode, setMode }: ModeSwitchProps) {
  return (
    <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 shadow-md"></div>
            <div>
              <div className="text-xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                SwingAI
              </div>
              <div className="text-xs text-cyan-600 font-medium">{mode === "player" ? "Player Lab" : "Coach Hub"}</div>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex gap-1.5 rounded-lg border border-gray-200 bg-gray-50 p-1.5">
            <button
              onClick={() => setMode("player")}
              className={`px-4 py-2 rounded-md font-semibold transition-all duration-300 flex items-center gap-2 ${
                mode === "player"
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <span>Player</span>
            </button>
            <button
              onClick={() => setMode("coach")}
              className={`px-4 py-2 rounded-md font-semibold transition-all duration-300 flex items-center gap-2 ${
                mode === "coach"
                  ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <span>Coach</span>
            </button>
          </div>

          <Link
            href="/"
            className="p-2.5 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors text-gray-600 hover:text-cyan-600"
          >
            <Home className="h-5 w-5" />
          </Link>
        </div>
      </div>
    </header>
  )
}
