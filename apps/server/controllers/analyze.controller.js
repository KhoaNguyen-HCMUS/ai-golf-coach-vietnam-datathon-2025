// apps/server/controllers/analyze.controller.js
import * as modelService from "../services/model.service.js";
import * as llmService from "../services/llm.service.js";

export const analyzeController = async (req, res) => {
  try {
    // Validate file was uploaded
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const videoPath = req.file.path;
    const timestamp = Date.now().toString();
    const hitIndex = 1;
    const clipId = `${timestamp}_hit_${hitIndex}`;

    // Step 1: Create segment object
    const segment = {
      clipId,
      timestamp,
      hitIndex,
      videoPath,
    };

    // Step 2: Get mock CV result from model.service
    const cvResult = modelService.getResult(segment);
    // console.log(`[Analyze] CV Result:`, cvResult);

    // Step 3: Call LLM to get HTML analysis
    const analysis = await llmService.processLLMAnalysis(cvResult);
    console.log(`[Analyze] Analysis received for ${clipId}`);

    // Step 4: Return response
    return res.status(200).json({
      message: "Analysis completed",
      data: analysis.analysisHTML,
    });
  } catch (error) {
    console.error("[Analyze] Error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
};