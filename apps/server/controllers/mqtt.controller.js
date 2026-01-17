import * as mqttService from "../services/mqtt.service.js";
import * as videoService from "../services/video.service.js";

export const startController = async (req, res) => {
  try {
    await mqttService.publishCmd("START");
    console.log("START controller called");
    res.status(200).json({ message: "START command published" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const stopController = async (req, res) => {
  try {
    await mqttService.publishCmd("STOP");
    console.log("STOP controller called");

    if (!req.file) {
      return res.status(200).json({
        message: "STOP command published (no video uploaded)",
      });
    }
    const hits = await waitForSessionData(5000);

    if (!hits) {
      return res.status(200).json({
        message: "No hits detected",
        videoUploaded: true,
      });
    }

    const videoPath = req.file.path;

    // Extract timestamp from originalname: video_TIMESTAMP_filename.mp4
    const timestampMatch = req.file.originalname.match(/video_(\d+)_/);
    const timestamp = timestampMatch
      ? timestampMatch[1]
      : Date.now().toString();
    console.log(
      `[Controller] 📤 Extracted timestamp: ${timestamp} from ${req.file.originalname}`
    );

    // Response IMMEDIATELY so FE gets timestamp quickly
    res.status(200).json({
      message: "Video uploaded. Processing started...",
      timestamp: timestamp,
      status: "processing",
    });

    // Process video ASYNC (don't block response)
    // Delay 2s to give FE time to subscribe
    setTimeout(async () => {
      try {
        console.log(
          `[Controller] 🎬 Processing video with timestamp: ${timestamp}`
        );
        console.log(`[Controller] 📍 Hits data:`, hits);
        // Extract hits array from response { start_time: ..., hits: [...] }
        const hitsArray = hits.hits || hits;
        const result = await videoService.processVideoWithHits(
          videoPath,
          hitsArray,
          timestamp
        );
        console.log(
          `[Controller] ✅ Completed - ${result.totalClips} clips processed`
        );
      } catch (error) {
        console.error(`[Controller] ❌ Error processing video:`, error);
      }
    }, 2000);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const stopWithoutVideoController = async (req, res) => {
  try {
    await mqttService.publishCmd("STOP");
    console.log("STOP without upload video");
    res.status(200).json({ message: "STOP without video uploaded" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

const waitForSessionData = (timeout) => {
  return new Promise((resolve) => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const data = mqttService.getLastHits();

      if (data) {
        clearInterval(interval);
        resolve(data);
      } else if (Date.now() - startTime > timeout) {
        clearInterval(interval);
        resolve(null);
      }
    }, 100);
  });
};
