import * as chatBotService from "../services/llm.service.js";

export const askController = async (req, res) => {
  try {
    const { history = [], message } = req.body || {};
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message is required" });
    }
    if (!Array.isArray(history)) {
      return res.status(400).json({ error: "history must be an array of strings" });
    }

    const result = await chatBotService.processChatHTML({ history, message });
    return res.status(200).json({ data: result.html, message: "Ask successful" });
  } catch (error) {
    console.error("[Chat] Error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
};