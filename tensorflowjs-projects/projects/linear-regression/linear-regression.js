const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const path = require("path");

const fs = require("fs");
const app = express.Router();

app.get("/read-csv", async (req, res) => {
  try {
    tf.kmeans;
    const parsedData = await parseCSV(); // Pass 3 as the second argument to take only the first 3 lines
    res.json(parsedData);
  } catch (error) {
    console.error("Error reading CSV:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

async function parseCSV() {
  const houseDataSet = tf.data.csv(
    "http://localhost:3000/utils/kc_house_data.csv"
  );

  const samplesDataset = houseDataSet.take(3);
  const dataArray = await samplesDataset.toArray();

  return houseDataSet;
}

function normalize(tensor, previousMin = null, previousMax = null) {
  const min = previousMin || tensor.min();
  const max = previousMax || tensor.max();

  const normalizedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normalizedTensor,
    min,
    max,
  };
}

function denormalize(tensor, min, max) {
  const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalizedTensor;
}

function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 1,
      useBias: true,
      activation: "linear",
      inputDim: 1,
    })
  );

  const optimizer = tf.train.sgd(0.1); // sgd with 0.1 Learning Rate

  model.compile({
    loss: "meanSquaredError",
    optimizer,
  });

  return model;
}

async function trainModel(model, trainingFeaturesTensor, trainingLabelsTensor) {
  return model.fit(trainingFeaturesTensor, trainingLabelsTensor, {
    batchSize: 32,
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch ${epoch} : loss =${log.loss}`),
    },
    onEpochBegin: async function () {
      await plotPredictionLine();
      const layer = model.getLayer(undefined, 0);
    },
  });
}
const storageID = "kc-house-regression";
const modelFolderPath = path.join(__dirname, "modelHistory"); // Define the folder path

async function saveModel(model) {
  // Save the model to a file inside the modelHistory folder
  const saveResults = await model.save(
    `file://${path.join(modelFolderPath, storageID)}`
  );
  return saveResults;
}

async function loadModel() {
  // Load the model from the saved file
  const model = await tf.loadLayersModel(
    `file://${path.join(modelFolderPath, storageID)}/model.json`
  );

  await plotPredictionLine();

  return model;
}

if (!fs.existsSync(modelFolderPath)) {
  fs.mkdirSync(modelFolderPath);
}

app.get("/load-model", async (req, res) => {
  try {
    // Load the model when needed (e.g., for inference)
    const loadedModel = await loadModel();
    console.log("Model loaded successfully");

    // Get information about layers
    const modelLayers = loadedModel.layers.map((layer) => ({
      name: layer.name,
      type: layer.getClassName(),
      outputShape: layer.outputShape,
      parameters: layer.countParams(),
    }));

    // Send the model information in JSON format
    res.json({
      status: "success",
      message: "Model loaded successfully",
      modelLayers: modelLayers,
    });
  } catch (error) {
    console.error("Error loading model:", error.message);
    res.status(500).json({ status: "error", error: "Internal server error" });
  }
});

// Check if the modelHistory folder exists, if not, create it
if (!fs.existsSync(modelFolderPath)) {
  fs.mkdirSync(modelFolderPath);
}

// Move these variables to a higher scope
let normalizedFeature;
let normalizedLabel;

app.get("/linear-regression", async (req, res) => {
  const houseDataSet = await parseCSV();

  const pointsDataset = houseDataSet.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  const points = await pointsDataset.toArray();

  if (points.length % 2 !== 0) {
    points.pop();
  }

  //shuffle
  tf.util.shuffle(points);

  //inputs
  const featureValues = points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  // outputs
  const labelValues = points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  normalizedFeature = normalize(featureTensor);
  normalizedLabel = normalize(labelTensor);

  //normalizedFeature.tensor.print();
  //normalizedLabel.tensor.print();

  const [trainingFeaturesTensor, testingFeaturesTensor] = tf.split(
    normalizedFeature.tensor,
    2
  );
  const [trainingLabelsTensor, testingLabelsTensor] = tf.split(
    normalizedLabel.tensor,
    2
  );

  trainingFeaturesTensor.print(true);

  model = createModel();
  await plotPredictionLine();

  const result = await trainModel(
    model,
    trainingFeaturesTensor,
    trainingLabelsTensor
  );
  console.log("train result :", result);

  //Save model after training :

  await saveModel(model);

  const trainingLoss = result.history.loss.pop();

  console.log(`Training set loss: ${trainingLoss}`);

  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation set loss: ${validationLoss}`);

  const lossTensor = model.evaluate(
    testingFeaturesTensor,
    trainingLabelsTensor
  );

  const loss = await lossTensor.dataSync();

  console.log(`Training set loss: ${loss}`);

  model.summary();

  denormalize(
    normalizedFeature.tensor,
    normalizedFeature.min,
    normalizedFeature.max
  ).print();

  return res.json(points);
});

app.get("/predict", async (req, res) => {
  try {
    const x = parseFloat(req.query.x);

    if (isNaN(x)) {
      res.status(400).json({
        error: "Invalid input. Please provide a valid number for 'x'.",
      });
      return;
    }

    const loadedModel = await loadModel();
    console.log("Model loaded successfully");

    let outputValueRounded;
    // Perform prediction
    const prediction = tf.tidy(() => {
      const inputTensor = tf.tensor1d([x]);
      // Assuming normalizedFeature, normalizedLabel are available
      const normalizedInput = normalize(
        inputTensor,
        normalizedFeature.min,
        normalizedFeature.max
      );
      const normalizedOutputTensor = loadedModel.predict(
        normalizedInput.tensor
      );
      const outputTensor = denormalize(
        normalizedOutputTensor,
        normalizedLabel.min,
        normalizedLabel.max
      );

      const outputValue = outputTensor.dataSync()[0];

      outputValueRounded = (outputValue / 1000).toFixed(0) * 1000;

      return outputValueRounded;
    });

    console.log("The predicted house price is", outputValueRounded);
    res.json({ prediction: outputValueRounded });
  } catch (error) {
    console.error("Error predicting house price:", error.message);
    res.status(500).json({ status: "error", error: "Internal server error" });
  }
});

app.get("/get-predicted-line", async (req, res) => {
  try {
    const predictedLineData = await getPredictedLineData();
    res.json({ predictedLineData });
  } catch (error) {
    console.error("Error getting predicted line data:", error.message);
    res.status(500).json({ status: "error", error: "Internal server error" });
  }
});

async function getPredictedLineData() {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs = tf.linspace(0, 1, 100);
    const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));

    const xs = denormalize(
      normalisedXs,
      normalizedFeature.min,
      normalizedFeature.max
    );
    const ys = denormalize(
      normalisedYs,
      normalizedLabel.min,
      normalizedLabel.max
    );

    return [xs.dataSync(), ys.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, index) => ({
    x: val,
    y: ys[index],
  }));

  return predictedPoints;
}

let model;
async function plotPredictionLine() {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs = tf.linspace(0, 1, 100);
    const normalisedYs = model.predict(normalisedXs.reshape([100, 1]));

    const xs = denormalize(
      normalisedXs,
      normalizedFeature.min,
      normalizedFeature.max
    );
    const ys = denormalize(
      normalisedYs,
      normalizedLabel.min,
      normalizedLabel.max
    );

    return [xs.dataSync(), ys.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, index) => ({
    x: val,
    y: ys[index],
  }));

  console.log("hii", predictedPoints);
  return predictedPoints;
}

module.exports = app;
