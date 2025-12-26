"use client"

import { useState } from "react"
import PlayerMode from "./components/PlayerMode"
import CoachMode from "./components/CoachMode"
import ModeSwitch from "./components/ModeSwitch"

export default function App() {
  const [mode, setMode] = useState<"player" | "coach">("player")

  return (
    <div className="min-h-screen bg-background text-foreground">
      <ModeSwitch mode={mode} setMode={setMode} />
      {mode === "player" ? <PlayerMode /> : <CoachMode />}
    </div>
  )
}
