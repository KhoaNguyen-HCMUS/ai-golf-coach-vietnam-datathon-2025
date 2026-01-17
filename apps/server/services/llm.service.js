import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

const PROMPT = `You are an expert professional golf coach. Analyze the detailed swing analysis data and provide coaching feedback in HTML format.

IMPORTANT: Use the provided analysis data (score, band, insights, features) as the foundation for your response. Focus on:
1. What the overall band score means for the golfer's skill level
2. The key strengths mentioned in the analysis  
3. The areas for improvement highlighted
4. Actionable recommendations based on the specific biomechanical and kinematic features

Return your analysis as clean HTML that can be directly inserted into a webpage. Use these elements:
- <h3> for section headings
- <p> for paragraphs
- <strong> for emphasis
- <ul> and <li> for lists
- Use colors: green (#4caf50) for good points, orange (#ff9800) for improvements, red (#f44336) for issues

Structure your response with these sections:
1. Overall Assessment (with confidence level and what band score means)
2. Key Strengths (reference the provided strengths)
3. Areas for Improvement (reference the provided weaknesses)
4. Important Features Analysis (highlight the most important features affecting the score)
5. Specific Recommendations (actionable drills and focus areas)

Make it professional, encouraging, and actionable. Be specific using the feature names and values provided.`;

const PROMPT1 = `You are an expert golf coach. Answer in clean HTML (no markdown fences). Use <h3>, <p>, <ul>, <li>, <strong>. Keep it concise, supportive, and actionable.`;

export const processLLMAnalysis = async (cvResult) => {
  console.log(`[LLM] Analyzing ${cvResult.clipId}`);

  // Build detailed feature text from the features array
  const topFeatures = (cvResult.features || [])
    .slice(0, 8)
    .map(
      (f) =>
        `- ${f.name} (${f.key}): ${f.value.toFixed(2)} ${f.unit} - ${
          f.evaluation
        } (${f.description})`
    )
    .join("\n");

  const strengthsList = (cvResult.insights?.strengths || [])
    .map((s) => `- ${s}`)
    .join("\n");

  const weaknessesList = (cvResult.insights?.weaknesses || [])
    .map((w) => `- ${w}`)
    .join("\n");

  const prompt = `${PROMPT}

Swing Analysis Results:
Score Band: ${cvResult.score} (Band Index: ${cvResult.band_index}/10)
Confidence: ${(cvResult.confidence * 100).toFixed(1)}%

Strengths:
${strengthsList || "No specific strengths identified"}

Areas for Improvement:
${weaknessesList || "No specific weaknesses identified"}

Key Features (importance weighted):
${topFeatures}

Probabilities by Band:
${Object.entries(cvResult.probabilities || {})
  .map(([band, prob]) => `- ${band}: ${(prob * 100).toFixed(1)}%`)
  .join("\n")}

Provide your professional coaching analysis in HTML format.`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const htmlAnalysis = response.text();

    console.log(
      "[LLM] HTML Analysis received:",
      htmlAnalysis.substring(0, 200) + "..."
    );

    // Clean HTML (remove markdown code blocks if present)
    let cleanHTML = htmlAnalysis
      .replace(/```html\n?/g, "")
      .replace(/```\n?/g, "")
      .trim();

    return {
      clipId: cvResult.clipId,
      timestamp: cvResult.timestamp,
      hitIndex: cvResult.hitIndex,
      videoPath: cvResult.videoPath,
      analysisHTML: cleanHTML, // HTML content
    };
  } catch (error) {
    console.error(`[LLM] Error:`, error.message);

    // Fallback HTML
    const fallbackHTML = `
            <h3 style="color: #f44336;">Analysis Failed</h3>
            <p>Unable to analyze swing at this time.</p>
            <p><strong>Error:</strong> ${error.message}</p>
        `;

    return {
      clipId: cvResult.clipId,
      timestamp: cvResult.timestamp,
      hitIndex: cvResult.hitIndex,
      analysisHTML: fallbackHTML,
    };
  }
};

export const processChatHTML = async ({ history = [], message }) => {
  const historyText = history.map((q, i) => `Q${i + 1}: ${q}`).join("\n");
  const prompt = `${PROMPT1}
Conversation history:
${historyText || "(no previous questions)"}

User question:
${message}

Return ONLY HTML (no code fences).`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    let html = response.text() || "";

    // Cleanup code fences if the model adds them
    html = html
      .replace(/```html\s*/gi, "")
      .replace(/```\s*/g, "")
      .trim();

    return { html };
  } catch (error) {
    console.error("[LLM] Chat error:", error.message);
    const fallback = `
      <h3 style="color:#f44336;">Chat Unavailable</h3>
      <p>Unable to generate a response right now.</p>
      <p><strong>Error:</strong> ${error.message}</p>
    `;
    return { html: fallback };
  }
};
