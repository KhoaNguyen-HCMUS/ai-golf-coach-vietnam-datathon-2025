import ffmpeg from 'fluent-ffmpeg'
import path from 'path'
import fs from 'fs'

const OUTPUT_BASE_DIR = './videos/output'
const OFFSET_MS = 1000 // 3 seconds offset

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

    const results = []
    
    for (let i = 0; i < hits.length; i++) {
        const hitRelativeTime = hits[i]
        const startTime = Math.max(0, (hitRelativeTime - OFFSET_MS) / 1000) // convert to seconds
        const duration = 2 // 1s before + 1s after
        
        const outputPath = path.join(outputDir, `hit_${i + 1}.mp4`)
        
        try {
            await cutVideoSegment(videoPath, startTime, duration, outputPath)
            results.push({
                hitIndex: i + 1,
                hitTime: hitRelativeTime,
                videoPath: outputPath
            })
            console.log(`Cut hit ${i + 1}: ${outputPath}`)
        } catch (error) {
            console.error(`Error cutting segment ${i + 1}:`, error)
        }
    }

    return {
        timestamp,
        outputDir,
        segments: results
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