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

    // Step 2: Get CV result from model.service
    const cvResult = await modelService.getResult(segment);
    console.log(`[Analyze] CV Result received`);

    // Step 3: Call LLM to get HTML analysis
    const analysis = await llmService.processLLMAnalysis(cvResult);
    //   const analysis = {
    //     analysisHTML: `
    //   <h3>Overall Assessment</h3>
    //   <p><strong>Score: 85/100</strong></p>
    //   <p>Excellent swing with great tempo and balance.</p>

    //   <h3>Technical Analysis</h3>
    //   <ul>
    //     <li><strong>Backswing:</strong> 87.5° - Great rotation</li>
    //     <li><strong>Hip Rotation:</strong> 45.2° - Good</li>
    //     <li><strong>Shoulder Rotation:</strong> 92.3° - Excellent</li>
    //     <li><strong>Club Speed:</strong> 145.3 km/h - Strong</li>
    //   </ul>

    //   <h3 style="color: #4caf50;">Strengths</h3>
    //   <ul>
    //     <li style="color: #4caf50;">✓ Perfect stance and alignment</li>
    //     <li style="color: #4caf50;">✓ Smooth backswing</li>
    //     <li style="color: #4caf50;">✓ Strong impact position</li>
    //   </ul>

    //   <h3 style="color: #ff9800;">Improvements</h3>
    //   <ul>
    //     <li style="color: #ff9800;">⚠ Hip rotation: aim for 50°</li>
    //     <li style="color: #ff9800;">⚠ Work on follow-through</li>
    //   </ul>

    //   <h3>Recommendations</h3>
    //   <ol>
    //     <li>Practice hip mobility drills</li>
    //     <li>Focus on swing consistency</li>
    //     <li>Keep up the great work!</li>
    //   </ol>
    // `
    //   }
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
