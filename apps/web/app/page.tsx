"use client"

import Link from "next/link"
import {
  ArrowRight,
  BarChart3,
  Users,
  Zap,
  Target,
  Brain,
  Video,
  CheckCircle2,
  Play,
  Smartphone,
  TrendingUp,
  Award,
  Sparkles,
} from "lucide-react"
import { useState } from "react"

export default function LandingPage() {
  const [hoveredCard, setHoveredCard] = useState<number | null>(null)

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-blue-50 to-white overflow-hidden">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-200/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/4 w-72 h-72 bg-cyan-200/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-gray-200/50 bg-white/80 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3 group">
              <div className="h-10 w-10 rounded-xl overflow-hidden shadow-lg shadow-blue-200 group-hover:shadow-blue-300 transition-all">
                <img src="/logo.png" alt="SwingAI Logo" className="h-full w-full object-contain" />
              </div>
              <div>
                <div className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                  SwingAI
                </div>
                <div className="text-xs text-blue-600/60 font-medium">Pro Coaching</div>
              </div>
            </div>
            <div className="hidden items-center gap-8 md:flex">
              <a href="#features" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors">
                Features
              </a>
              <a href="#pricing" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors">
                Pricing
              </a>
              <a href="#" className="text-sm font-medium text-gray-600 hover:text-blue-600 transition-colors">
                Enterprise
              </a>
            </div>
            <div className="flex items-center gap-3">
              <button className="hidden sm:block rounded-lg px-4 py-2 text-sm font-semibold text-gray-700 hover:text-blue-600 transition-colors">
                Sign In
              </button>
              <Link
                href="/app"
                className="rounded-lg bg-gradient-to-r from-blue-500 to-cyan-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-blue-300/40 hover:shadow-blue-300/60 transition-all hover:scale-105 active:scale-95"
              >
                Try Free
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-24 pb-32 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-5xl relative z-10">
          {/* Badge */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex items-center gap-2 rounded-full border border-blue-300/40 bg-blue-100/40 px-4 py-2 backdrop-blur-sm">
              <Sparkles className="h-4 w-4 text-blue-600 animate-pulse" />
              <span className="text-xs sm:text-sm font-semibold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                AI-Powered Golf Analysis Platform
              </span>
            </div>
          </div>

          {/* Main headline */}
          <h1 className="text-center mb-6 text-5xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-tight">
            <span className="block text-gray-900">Master Your</span>
            <span className="block bg-gradient-to-r from-blue-600 via-cyan-600 to-blue-600 bg-clip-text text-transparent">
              Golf Swing
            </span>
            <span className="block text-gray-900">with AI Coaching</span>
          </h1>

          {/* Subheading */}
          <p className="mx-auto mb-10 max-w-2xl text-center text-lg text-gray-600 leading-relaxed">
            Real-time biomechanics analysis, expert feedback, and personalized coaching. Transform your game with
            precision insights powered by advanced AI technology.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              href="/app"
              className="group flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-8 py-4 font-semibold text-white shadow-xl shadow-blue-200/40 hover:shadow-blue-300/60 transition-all hover:scale-105 active:scale-95 w-full sm:w-auto justify-center"
            >
              Start Free Trial <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <button className="flex items-center gap-2 rounded-xl border border-blue-300/50 bg-white/50 backdrop-blur-sm px-8 py-4 font-semibold text-gray-700 hover:bg-gray-50 hover:border-blue-400/50 transition-all w-full sm:w-auto justify-center">
              <Play className="h-5 w-5" /> Watch Demo
            </button>
          </div>

          {/* Trust badges */}
          <div className="flex flex-wrap justify-center gap-6 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-blue-600" />
              <span>50K+ Active Golfers</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-blue-600" />
              <span>98% Accuracy Rate</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-blue-600" />
              <span>4.9 Star Rating</span>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-16 px-4 sm:px-6 lg:px-8 border-y border-gray-200/50 bg-gradient-to-r from-blue-50/50 to-cyan-50/50">
        <div className="mx-auto max-w-7xl">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: "50K+", label: "Active Players", icon: Users },
              { value: "2M+", label: "Swings Analyzed", icon: TrendingUp },
              { value: "98%", label: "Accuracy", icon: Target },
              { value: "4.9★", label: "User Rating", icon: Award },
            ].map((stat, i) => {
              const IconComponent = stat.icon
              return (
                <div key={i} className="text-center group">
                  <IconComponent className="h-8 w-8 text-blue-600 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                  <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-600 mt-2">{stat.label}</div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-24 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="mb-20 text-center">
            <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-4">Powerful Features for Champions</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Everything you need to analyze, improve, and master your golf swing
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: Video,
                title: "Instant Video Analysis",
                description:
                  "Upload your swing and get comprehensive AI analysis in seconds with frame-by-frame breakdown",
                color: "blue",
              },
              {
                icon: Brain,
                title: "Biomechanics Breakdown",
                description:
                  "8 keyframes with metrics on swing plane, rotation, hip clearance, and alignment precision",
                color: "cyan",
              },
              {
                icon: Users,
                title: "Expert Coaching",
                description: "Get personalized feedback from certified golf coaches with expert-in-the-loop validation",
                color: "blue",
              },
              {
                icon: BarChart3,
                title: "Progress Tracking",
                description: "Monitor improvements with detailed analytics, trends, and performance metrics over time",
                color: "cyan",
              },
              {
                icon: Target,
                title: "Coach Dashboard",
                description:
                  "Review student submissions, validate AI analysis, and provide expert corrections efficiently",
                color: "blue",
              },
              {
                icon: Zap,
                title: "AI Chat Assistant",
                description: "Ask questions and get instant AI-powered golf coaching advice tailored to your swing",
                color: "cyan",
              },
            ].map((feature, i) => {
              const IconComponent = feature.icon
              const isHovered = hoveredCard === i
              const bgColor = feature.color === "blue" ? "bg-blue-100/50" : "bg-cyan-100/50"
              const borderColor = feature.color === "blue" ? "border-blue-300/50" : "border-cyan-300/50"
              const hoverBorder = feature.color === "blue" ? "hover:border-blue-400/70" : "hover:border-cyan-400/70"
              const iconBg = feature.color === "blue" ? "bg-blue-100" : "bg-cyan-100"
              const iconColor = feature.color === "blue" ? "text-blue-600" : "text-cyan-600"

              return (
                <div
                  key={i}
                  onMouseEnter={() => setHoveredCard(i)}
                  onMouseLeave={() => setHoveredCard(null)}
                  className={`group relative rounded-2xl border transition-all duration-300 p-8 overflow-hidden ${borderColor} ${hoverBorder} ${bgColor} hover:shadow-lg hover:shadow-blue-100`}
                >
                  <div className="relative z-10">
                    <div
                      className={`mb-4 h-14 w-14 rounded-xl ${iconBg} flex items-center justify-center group-hover:scale-110 transition-transform`}
                    >
                      <IconComponent className={`h-7 w-7 ${iconColor}`} />
                    </div>
                    <h3 className="mb-3 text-xl font-bold text-gray-900 group-hover:text-transparent group-hover:bg-gradient-to-r group-hover:from-blue-600 group-hover:to-cyan-600 group-hover:bg-clip-text transition-all">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="relative py-24 px-4 sm:px-6 lg:px-8 border-t border-gray-200/50 bg-gray-50/50">
        <div className="mx-auto max-w-7xl">
          <div className="mb-20 text-center">
            <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-4">Simple Pricing</h2>
            <p className="text-lg text-gray-600">Choose the perfect plan for your golf journey</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                name: "Starter",
                price: "$9",
                description: "Perfect for practicing golfers",
                features: [
                  "10 video analyses/month",
                  "AI feedback included",
                  "Basic analytics",
                  "Mobile app access",
                  "Community forum",
                ],
                popular: false,
              },
              {
                name: "Professional",
                price: "$29",
                description: "For serious golfers and coaches",
                features: [
                  "Unlimited video analyses",
                  "AI feedback + Expert coaching",
                  "Advanced analytics",
                  "Coach mode access",
                  "Priority support",
                  "Progress reports",
                ],
                popular: true,
              },
              {
                name: "Enterprise",
                price: "Custom",
                description: "For golf academies and teams",
                features: [
                  "Unlimited everything",
                  "Priority support",
                  "Custom integrations",
                  "Dedicated account manager",
                  "Team management",
                  "API access",
                ],
                popular: false,
              },
            ].map((plan, i) => (
              <div
                key={i}
                className={`group relative rounded-2xl border p-8 transition-all duration-300 ${plan.popular
                  ? "border-blue-400/70 bg-gradient-to-b from-white to-blue-50/50 ring-2 ring-blue-200/50 shadow-lg shadow-blue-100/50"
                  : "border-gray-200/70 bg-white hover:border-gray-300 hover:shadow-md hover:shadow-gray-100"
                  }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-600 px-4 py-1.5 text-xs font-bold text-white shadow-lg shadow-blue-200">
                    <Sparkles className="h-3 w-3" />
                    MOST POPULAR
                  </div>
                )}

                <div className="mb-6">
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                  <p className="text-sm text-gray-600">{plan.description}</p>
                </div>

                <div className="mb-8">
                  <div className="flex items-baseline gap-1">
                    <span className="text-5xl font-bold text-gray-900">{plan.price}</span>
                    {plan.price !== "Custom" && <span className="text-gray-600">/month</span>}
                  </div>
                </div>

                <button
                  className={`w-full rounded-xl font-semibold py-3.5 mb-8 transition-all active:scale-95 ${plan.popular
                    ? "bg-gradient-to-r from-blue-500 to-cyan-600 text-white shadow-lg shadow-blue-200/40 hover:shadow-blue-300/60"
                    : "border border-gray-300 text-gray-900 hover:bg-gray-50 bg-white"
                    }`}
                >
                  {plan.popular ? "Start Now" : plan.price === "Custom" ? "Contact Sales" : "Get Started"}
                </button>

                <div className="space-y-4">
                  {plan.features.map((feature, j) => (
                    <div key={j} className="flex items-start gap-3">
                      <CheckCircle2
                        className={`h-5 w-5 mt-0.5 flex-shrink-0 ${plan.popular ? "text-blue-600" : "text-gray-400"}`}
                      />
                      <span className="text-sm text-gray-700">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-24 px-4 sm:px-6 lg:px-8 border-t border-gray-200/50">
        <div className="mx-auto max-w-4xl text-center">
          <h2 className="text-4xl sm:text-5xl font-bold text-gray-900 mb-6">Ready to Transform Your Game?</h2>
          <p className="text-lg text-gray-600 mb-12">
            Join thousands of golfers improving their swing with AI coaching today
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/app"
              className="group flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-600 px-8 py-4 font-semibold text-white shadow-xl shadow-blue-200/40 hover:shadow-blue-300/60 transition-all hover:scale-105 active:scale-95"
            >
              Start Free Trial <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <button className="flex items-center justify-center gap-2 rounded-xl border border-blue-300/50 bg-white/50 px-8 py-4 font-semibold text-gray-700 hover:bg-white transition-all">
              <Smartphone className="h-5 w-5" /> Use Mobile
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200/50 bg-white/80 backdrop-blur-sm py-12 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Product</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Features
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Pricing
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Updates
                </a>
              </div>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Company</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <a href="#" className="hover:text-blue-600 transition-colors">
                  About
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Blog
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Careers
                </a>
              </div>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Resources</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Documentation
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Help Center
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  API
                </a>
              </div>
            </div>
            <div>
              <h4 className="font-bold text-gray-900 mb-4">Legal</h4>
              <div className="space-y-2 text-sm text-gray-600">
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Privacy
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Terms
                </a>
                <a href="#" className="hover:text-blue-600 transition-colors">
                  Contact
                </a>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-200 pt-8 text-center text-sm text-gray-600">
            <p>© 2025 SwingAI Pro. All rights reserved. Transforming golf coaching through AI.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
