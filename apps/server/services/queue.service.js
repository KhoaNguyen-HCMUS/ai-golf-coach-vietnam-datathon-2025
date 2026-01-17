import * as llmService from "./llm.service.js";
import * as websocketService from "./websocket.service.js";
import * as modelService from "./model.service.js";
import fs from "fs";

const queue = [];
let isProcessing = false;

export const addToQueue = (segment) => {
  queue.push(segment);
  console.log(`[Queue] Added ${segment.clipId} (size: ${queue.length})`);

  if (!isProcessing) {
    processQueue();
  }
};

export const processQueue = async () => {
  if (queue.length === 0) {
    isProcessing = false;
    console.log("[Queue] Empty");
    return;
  }

  isProcessing = true;
  const segment = queue.shift();
  const sendToClient = websocketService.sendToClient;

  console.log(`[Queue] Processing ${segment.clipId}`);

  // Import các function cần thiết
  // const { processCVModel } = await import('./cv.service.js')
  // const cvResult = modelService.getResult(segment);
  // const analysis = await llmService.processLLMAnalysis(cvResult);
  const analysis = {
    clipId: segment.clipId,
    timestamp: segment.timestamp,
    hitIndex: segment.hitIndex,
    analysisHTML: `
    <h3>Overall Assessment</h3>
    <p><strong>Score: 85/100</strong></p>
    <p>Solid swing with good tempo and balance. Club speed at 145.3 km/h is excellent.</p>
    
    <h3>Strengths</h3>
    <ul>
      <li style="color: #4caf50;">✓ Excellent stance and alignment</li>
      <li style="color: #4caf50;">✓ Smooth backswing rotation (87.5°)</li>
      <li style="color: #4caf50;">✓ Strong impact position</li>
    </ul>
    
    <h3>Areas to Improve</h3>
    <ul>
      <li style="color: #ff9800;">⚠ Hip rotation could be better (45.2° → aim for 50°)</li>
      <li style="color: #ff9800;">⚠ Work on follow-through consistency</li>
    </ul>
    
    <h3>Recommendations</h3>
    <ol>
      <li>Practice hip mobility drills 3x weekly</li>
      <li>Focus on swing repeatability</li>
      <li>Great job! Keep practicing!</li>
    </ol>
  `,
  };
  const videoBuffer = fs.readFileSync(segment.videoPath);
  const videoBase64 = videoBuffer.toString("base64");

  sendToClient(segment.timestamp, {
    type: "completed",
    data: {
      ...analysis,
      video: {
        base64: videoBase64,
        mimeType: "video/mp4",
        filename: `hit_${segment.hitIndex}.mp4`,
        size: videoBuffer.length,
      },
    },
  });

  console.log(`Result: ${JSON.stringify(analysis)}`);
  console.log(`[Queue] Completed ${segment.clipId}`);
  processQueue();
};

export const getQueueSize = () => queue.length;
