import * as mqttService from "../services/mqtt.service.js";
import * as videoService from "../services/video.service.js";

export const startController = async (req, res) => {
  try {
    await mqttService.publishCmd("START");
    res.status(200).json({ message: "START command published" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

export const stopController = async (req, res) => {
  try {
    await mqttService.publishCmd("STOP");

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
    await videoService.processVideoWithHits(videoPath, hits);

    res.status(200).json({
      message: "Video processed successfully",
    });
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
