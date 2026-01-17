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

  try {
    const cvResult = await modelService.getResult(segment);
    const analysis = await llmService.processLLMAnalysis(cvResult);

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

    console.log(`[Queue] Completed ${segment.clipId}`);
  } catch (error) {
    console.error(`[Queue] Error processing ${segment.clipId}:`, error.message);
    sendToClient(segment.timestamp, {
      type: "error",
      data: {
        clipId: segment.clipId,
        error: error.message,
      },
    });
  }

  processQueue();
};

export const getQueueSize = () => queue.length;
