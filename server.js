const express = require("express");
const bodyParser = require("body-parser");
const axios = require("axios");
const tf = require("@tensorflow/tfjs-node");
const app = express();
const fs = require("fs");

const port = 3001;
app.use(bodyParser.json());

const OPENWEATHERMAP_API_KEY = "008d1ffcd506450f980fdab36e070351";
const OPENWEATHERMAP_API_ENDPOINT =
  "http://api.openweathermap.org/data/2.5/weather";
const MODELS_DIR = "./weather_models";
const LOG_FILE = "./weather_prediction_log.txt";

const categories = [
  "Sunny",
  "Rainy",
  "Cloudy",
  "Snowy",
  "Windy",
  "Foggy",
  "Unknown",
  "Variable1",
  "Variable2",
  "Dangerous",
];

const numberOfCategories = categories.length;

let trainedModel;

async function preprocessWeatherData(weatherData) {
  try {
    if (
      !weatherData ||
      !weatherData.main ||
      typeof weatherData.main.temp === "undefined" ||
      !weatherData.wind ||
      typeof weatherData.wind.speed === "undefined"
    ) {
      throw new Error("Invalid or incomplete weather data format");
    }

    const temperature = weatherData.main.temp;
    const humidity = weatherData.main.humidity;
    const windSpeed = weatherData.wind.speed;
    const weatherCategory = "Unknown"; // Replace with actual logic to determine weather category

    return { temperature, humidity, windSpeed, weatherCategory };
  } catch (error) {
    throw new Error(`Error preprocessing weather data: ${error.message}`);
  }
}

async function trainModel(xs, ys, modelConfig) {
  const model = tf.sequential();

  const lstmUnits = 128;
  model.add(
    tf.layers.lstm({
      units: lstmUnits,
      inputShape: [xs.shape[1], xs.shape[2]],
      returnSequences: false,
    })
  );

  model.add(
    tf.layers.dense({
      units: numberOfCategories,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  model.summary();

  const ysOneHot = tf.oneHot(tf.argMax(ys, 1), numberOfCategories);

  await model.fit(xs, ysOneHot, {
    epochs: modelConfig.fit.epochs,
    metrics: ["accuracy"],
  });

  await model.save(`file://${__dirname}/weather_models/${modelConfig.name}`);
  return model;
}

app.post("/predictFuture", async (req, res) => {
  try {
    const city = req.body.city;
    const futureDate = req.body.futureDate;

    const trainingData = await getHistoricalWeatherData(city);

    const xs = tf.tensor3d(
      trainingData.map((data) => [
        [data.input.temperature],
        [data.input.humidity],
        [data.input.windSpeed],
      ]),
      [trainingData.length, 3, 1]
    );
    const ys = tf.tensor2d(
      trainingData.map((data) => {
        const categoryIndex = categories.indexOf(data.output.category);
        return Array.from({ length: categories.length }, (_, i) =>
          i === categoryIndex ? 1 : 0
        );
      }),
      [trainingData.length, numberOfCategories]
    );

    const modelConfig = {
      name: "weather_model",
      fit: { epochs: 100 },
    };

    const trainedModel = await trainModel(xs, ys, modelConfig);

    const currentWeatherData = await getDynamicWeatherData(city);
    const { temperature, humidity, windSpeed } = await preprocessWeatherData(
      currentWeatherData
    );

    const inputTensor = tf.tensor3d([[[temperature], [humidity], [windSpeed]]]);

    const prediction = await trainedModel.predict(inputTensor);

    const result = {
      category: categories[prediction.argMax().dataSync()[0]],
      windSpeed: prediction.dataSync()[0], // Modify based on your actual output structure
      additionalVariables: {}, // Modify based on your actual output structure
    };

    // Log the prediction
    const logEntry = `${new Date().toISOString()} - City: ${city}, Future Date: ${futureDate}, Category: ${
      result.category
    }, Wind Speed: ${result.windSpeed}, Additional Variables: ${JSON.stringify(
      result.additionalVariables
    )}\n`;

    fs.appendFile(LOG_FILE, logEntry, (err) => {
      if (err) {
        console.error("Error writing to log file:", err.message);
      }
    });

    res.json({
      result: `Weather prediction for ${city} on future date ${futureDate}: Category: ${
        result.category
      }, Wind Speed: ${
        result.windSpeed
      }, Additional Variables: ${JSON.stringify(result.additionalVariables)}`,
    });
  } catch (error) {
    console.error("Error predicting the future:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

async function getDynamicWeatherData(city) {
  try {
    const response = await axios.get(OPENWEATHERMAP_API_ENDPOINT, {
      params: {
        q: city,
        appid: OPENWEATHERMAP_API_KEY,
      },
    });

    const weatherData = response.data;

    console.log("Full Weather Data:", weatherData); // Log the complete weather data

    if (
      !weatherData.main ||
        !weatherData.main.temp ||
        !weatherData.wind ||
        !weatherData.wind.speed
      ) {
        console.error("Invalid or incomplete weather data format");
        throw new Error("Invalid or incomplete weather data format");
      }

      return {
        temperature: weatherData.main.temp,
      humidity: weatherData.main.humidity,
      windSpeed: weatherData.wind.speed,
    };
  } catch (error) {
    console.error(
      `Error fetching or processing weather data: ${error.message}`
    );
    throw new Error(
      `Error fetching or processing weather data: ${error.message}`
    );
  }
}
async function getHistoricalWeatherData(city) {
  try {
    // Fetch historical weather data from a hypothetical database
    const historicalData = await WeatherDatabase.find({ city: city });
    
    // Convert historical data to the required format
    const trainingData = historicalData.map((data) => ({
      input: {
        temperature: data.temperature,
        humidity: data.humidity,
        windSpeed: data.windSpeed,
        // Add other properties as needed
      },
      output: { category: data.weatherCategory },
    }));

    return trainingData;
  } catch (error)   {
    console.error(`Error fetching historical weather data: ${error.message}`);
    throw new Error(`Error fetching historical weather data: ${error.message}`);
  }
}

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
