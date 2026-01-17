import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

const PROMPT = `You are an expert golf coach. Analyze this swing data and provide detailed feedback in HTML format.

Return your analysis as clean HTML that can be directly inserted into a webpage. Use these elements:
- <h3> for section headings
- <p> for paragraphs
- <strong> for emphasis
- <ul> and <li> for lists
- Use colors: green (#4caf50) for good points, orange (#ff9800) for improvements, red (#f44336) for issues

Structure your response with these sections:
1. Overall Assessment (with score out of 100)
2. Technical Analysis (stance, backswing, impact, follow-through)
3. Key Strengths
4. Areas for Improvement
5. Specific Recommendations

Make it professional, encouraging, and actionable.`;

const PROMPT1 = `You are an expert golf coach. Answer in clean HTML (no markdown fences). Use <h3>, <p>, <ul>, <li>, <strong>. Keep it concise, supportive, and actionable.`;

export const processLLMAnalysis = async (cvResult) => {
  console.log(`[LLM] Analyzing ${cvResult.clipId}`);

  const prompt = `${PROMPT}

Golf Swing Data:
- Backswing Angle: ${cvResult.tracking.person.pose_angles.backswing}°
- Hip Rotation: ${cvResult.tracking.person.pose_angles.hip_rotation}°
- Shoulder Rotation: ${cvResult.tracking.person.pose_angles.shoulder_rotation}°
- Club Speed: ${cvResult.tracking.club.speed_kmh} km/h
- Ball Speed: ${cvResult.tracking.ball.speed_kmh} km/h
- Launch Angle: ${cvResult.tracking.ball.launch_angle}°

Provide your professional analysis in HTML format.`;

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const htmlAnalysis = response.text();
//     const htmlAnalysis = `
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
// `;

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
  //   let html = `
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
    
  //   <h3 style="color: #ff9800;">Improv  ements</h3>
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
