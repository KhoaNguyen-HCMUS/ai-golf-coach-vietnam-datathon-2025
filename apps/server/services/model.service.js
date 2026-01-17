import axios from "axios";
import FormData from "form-data";
import fs from "fs";

const MODEL_API_URL = process.env.MODEL_API_URL;

export const getResult = async (segment) => {
  try {
    console.log(
      `[ModelService] Sending video to predict: ${segment.videoPath}`
    );

    // Read video file
    const videoStream = fs.createReadStream(segment.videoPath);

    // Create FormData
    const formData = new FormData();
    formData.append("file", videoStream, "video.mp4");

    // Call model API
    const response = await axios.post(`${MODEL_API_URL}/predict`, formData, {
      headers: formData.getHeaders(),
      timeout: 300000, // 5 minutes timeout for video processing
    });

    const modelResult = response.data;

    // Return enriched result with segment info and model data
    return {
      clipId: segment.clipId,
      timestamp: segment.timestamp,
      hitIndex: segment.hitIndex,
      videoPath: segment.videoPath,
      score: modelResult.score,
      band_index: modelResult.band_index,
      confidence: modelResult.confidence,
      probabilities: modelResult.probabilities,
      insights: modelResult.insights,
      features: modelResult.features,
    };
  } catch (error) {
    console.error(
      `[ModelService] Error processing video ${segment.videoPath}:`,
      error.message
    );

    // Return fallback result on error
    return {
      clipId: segment.clipId,
      timestamp: segment.timestamp,
      hitIndex: segment.hitIndex,
      videoPath: segment.videoPath,
      error: error.message,
      score: "error",
      insights: {
        strengths: ["Unable to analyze swing"],
        weaknesses: [],
      },
      features: [],
    };
  }
};
