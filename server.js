const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const fs = require("fs").promises;
const tf = require("@tensorflow/tfjs-node");
const app = express();

const port = 3001;
app.use(bodyParser.json());

const OPENWEATHERMAP_API_KEY = "008d1ffcd506450f980fdab36e070351";
const OPENWEATHERMAP_API_ENDPOINT =
  "http://api.openweathermap.org/data/2.5/weather";
const MODELS_DIR = "./weather_models";
const LOG_FILE = "./weather_prediction_log.txt";

async function fetchWeatherData(city) {
  try {
    const response = await axios.get(OPENWEATHERMAP_API_ENDPOINT, {
      params: {
        q: city,
        appid: OPENWEATHERMAP_API_KEY,
      },
    });
    const weatherData = response.data;
    return weatherData;
  } catch (error) {
    console.error("Error fetching weather data:", error.message);
    throw error;
  }
}

async function preprocessWeatherData(weatherData) {
  try {
    const { main, weather } = weatherData;
    const { temp, humidity } = main;

    const disaster = weather.some(
      (condition) => condition.main.toLowerCase() === "thunderstorm"
    );

    return {
      temperature: temp,
      humidity,
      disaster,
    };
  } catch (error) {
    console.error("Error preprocessing weather data:", error.message);
    throw error;
  }
}

async function trainModel(xs, ys, modelConfig) {
  const model = tf.sequential();
  model.add(tf.layers.dense(modelConfig.denseLayer));
  model.compile(modelConfig.compile);

  await model.fit(xs, ys, modelConfig.fit);

  await model.save(`file://${__dirname}/weather_models/${modelConfig.name}`);
  return model;
}

async function predictDisaster(model, inputData) {
  const inputTensor = tf.tensor2d([inputData]);
  const prediction = model.predict(inputTensor).dataSync()[0];
  return prediction >= 0.5 ? "Disaster" : "No Disaster";
}

async function setupModel(city, modelConfig) {
  try {
    const weatherData = await fetchWeatherData(city);
    const { temperature, humidity, disaster } = await preprocessWeatherData(
      weatherData
    );
    const xs = tf.tensor2d([[temperature, humidity]]);
    const ys = tf.tensor1d([disaster ? 1 : 0]);

    const trainedModel = await trainModel(xs, ys, modelConfig);

    return trainedModel;
  } catch (error) {
    console.error("Error setting up model:", error.message);
    throw error;
  }
}

(async () => {
  try {
    const city = "London";
    const modelConfig = {
      name: "disaster_model",
      denseLayer: { units: 1, inputShape: [2], activation: "sigmoid" },
      compile: {
        optimizer: "sgd",
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
      },
      fit: { epochs: 100 },
    };

    const trainedModel = await setupModel(city, modelConfig);

    app.post("/predict", async (req, res) => {
      try {
        const { city, temperature, humidity } = req.body;

        if (!city || !temperature || !humidity) {
          return res.status(400).json({
            error:
              "City, temperature, and humidity are required in the request body.",
          });
        }

        const trainedModel = await setupModel(city, modelConfig);

        const inputData = [temperature, humidity];
        const predictedOutcome = await predictDisaster(trainedModel, inputData);
        const chartData = {
          labels: ["No Disaster", "Disaster"],
          datasets: [
            {
              label: "Prediction Result",
              data: [1 - predictedOutcome, predictedOutcome], // Assuming predictedOutcome is a probability between 0 and 1
              backgroundColor: ["green", "red"],
            },
          ],
        };
        const logEntry = `
                    [${new Date().toISOString()}] City: ${city}, 
                    Input: [Temperature: ${temperature}, 
                        Humidity: ${humidity}], 
                    Result Predicted Outcome: ${predictedOutcome}\n`;
        await fs.appendFile(LOG_FILE, logEntry);

        res.json({
          result: `Outcome Prediction for ${city}: ${predictedOutcome}`,
          log: "Result logged to weather_prediction_log.txt.",
          chartData: chartData,
        });
      } catch (error) {
        console.error("Error during prediction:", error.message);
        res.status(500).json({ error: "Internal Server Error" });
      }
    });

    app.listen(port, () => {
      console.log(`Server is running at http://localhost:${port}`);
    });
  } catch (error) {
    console.error("Error during initialization:", error.message);
  }
})();

function runAutomaticPrediction() {
  setInterval(async () => {
    try {
      // Default city for automatic prediction based on weather
      const autoCity = "Paris";
      const modelConfig = {
        name: "disaster_model",
        denseLayer: { units: 1, inputShape: [2], activation: "sigmoid" },
        compile: {
          optimizer: "sgd",
          loss: "binaryCrossentropy",
          metrics: ["accuracy"],
        },
        fit: { epochs: 100 },
      };

      // Set up the model for automatic prediction
      const trainedModel = await setupModel(autoCity, modelConfig);

      // Fetch current weather data for the default city
      const weatherData = await fetchWeatherData(autoCity);
      const { temperature, humidity, disaster } = await preprocessWeatherData(
        weatherData
      );
      const autoInputData = [temperature, humidity];

      // Predict the outcome automatically based on weather results
      const autoPredictedOutcome = await predictDisaster(
        trainedModel,
        autoInputData
      );

      // Create a new table for automatic prediction based on weather results
      const autoPredictionWeatherTable = {
        city: autoCity,
        temperature: temperature,
        humidity: humidity,
        weatherDisaster: disaster,
        predictedOutcome: autoPredictedOutcome,
      };

      // Log the automatic prediction result
      const logEntry = `
                  [${new Date().toISOString()}] Auto Prediction - City: ${autoCity}, 
                  Input: [Temperature: ${temperature}, 
                      Humidity: ${humidity}], 
                  Result Predicted Outcome: ${autoPredictedOutcome}\n`;
      await fs.appendFile(LOG_FILE, logEntry);

      console.log("Automatic prediction logged:", autoPredictionWeatherTable);
    } catch (error) {
      console.error(
        "Error during automatic prediction based on weather:",
        error.message
      );
    }
  }, 60000); // Run every 1 minute (60 seconds)
}

runAutomaticPrediction();
