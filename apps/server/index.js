import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import * as mqttService from './services/mqtt.service.js';
import mqttRouter from './routes/mqtt.route.js';

const app = express();
const PORT = process.env.PORT || 5000;
const HOST = '10.20.121.231';

app.use(cors());
app.use(express.json());

app.use('/mqtt', mqttRouter);

app.get('/', (req, res) => {
  res.json({ message: 'Hello world!' });
});

app.listen(PORT, HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
  mqttService.connectMqtt();
});


