import { Router } from "express";
import multer from "multer";
import { analyzeController } from "../controllers/analyze.controller.js";

const storage = multer.diskStorage({
  destination: "./videos/Input/",
  filename: (req, file, cb) => {
    cb(null, `analyze_${Date.now()}_${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB max
});

const router = Router();

router.post("/", upload.single("video"), analyzeController);

export default router;