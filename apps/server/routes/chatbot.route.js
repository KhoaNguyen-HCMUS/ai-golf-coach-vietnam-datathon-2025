import { Router } from "express";
import { askController } from "../controllers/chatbot.controller.js";

const router = Router();

router.post("/ask", askController);

export default router;