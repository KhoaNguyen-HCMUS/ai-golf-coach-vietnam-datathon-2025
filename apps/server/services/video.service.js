import ffmpeg from 'fluent-ffmpeg'
import path from 'path'
import fs from 'fs'
import { EventEmitter } from 'events'
import { addToQueue } from './queue.service.js'

const OUTPUT_BASE_DIR = './videos/Output'
const OFFSET_MS = 1000 // 3 seconds offset

export const videoEventEmitter = new EventEmitter()

export const processVideoWithHits = async (videoPath, hits) => {
    if (!hits || hits.length === 0) {
        console.log('No hits to process')
        return []
    }

    const filename = path.basename(videoPath)
    const timestampMatch = filename.match(/video_(\d+)_/)
    const timestamp = timestampMatch ? timestampMatch[1] : Date.now().toString()
    
    const outputDir = path.join(OUTPUT_BASE_DIR, timestamp)
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true })
        console.log(`Created output directory: ${outputDir}`)
    }

    const segments = []
    
    for (let i = 0; i < hits.length; i++) {
        const hitRelativeTime = hits[i]
        const startTime = Math.max(0, (hitRelativeTime - OFFSET_MS) / 1000) // convert to seconds
        const duration = 2 // 1s before + 1s after
        
        const clipId = `${timestamp}_hit_${i + 1}`
        const outputPath = path.join(outputDir, `hit_${i + 1}.mp4`)
        
        try {
            await cutVideoSegment(videoPath, startTime, duration, outputPath)
            const segment = {
                clipId,
                timestamp,
                hitIndex: i + 1,
                hitTime: hitRelativeTime,
                videoPath: outputPath,
                status: 'cut_completed'
            }

            segments.push(segment)
            console.log(`Cut hit ${i + 1}: ${outputPath}`)

            // videoEventEmitter.emit('clip:ready', segment)
            addToQueue(segment)
        } catch (error) {
            console.error(`Error cutting segment ${i + 1}:`, error)
        }
    }

    return {
        timestamp,
        outputDir,
        totalClips: segments.length,
        segments: segments
    }
}

const cutVideoSegment = (inputPath, startTime, duration, outputPath) => {
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .setStartTime(startTime)
            .setDuration(duration)
            .output(outputPath)
            .on('end', () => resolve(outputPath))
            .on('error', (err) => reject(err))
            .run()
    })
}