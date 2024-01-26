const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

// Function to generate dummy data for training and testing
function createDummyData(numPoints, numFeatures, numLabels) {
  // Generate random normal features and round uniform labels
  const features = tf.randomNormal([numPoints, numFeatures]);
  const labels = tf.round(tf.randomUniform([numPoints, numLabels]));
  return { features, labels };
}

// Function to create a TensorFlow.js model
function createModel(numFeatures) {
  // Create a sequential model with dense layers, ReLU activation, and sigmoid output
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "relu",
      inputShape: [numFeatures],
    })
  );
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  // Compile the model with optimizer, loss, and metrics
  model.compile({
    optimizer: "adam",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

// Function to train the TensorFlow.js model asynchronously
async function trainModel(model, features, labels) {
  // Dummy training call with 10 epochs
  await model.fit(features, labels, { epochs: 10 });
}

// Function to handle model predictions asynchronously
async function predict(model, inputData) {
  // Convert input data to a 2D tensor
  const tensorData = tf.tensor2d(inputData);
  // Use the model to make predictions
  return model.predict(tensorData);
}

// Function to save the trained model as JSON
function saveModelAsJSON(model, modelSavePath) {
  const modelJSON = model.toJSON();
  fs.writeFileSync(modelSavePath, JSON.stringify(modelJSON));
}

// Function to ensure the directory exists before saving the model
function ensureDirectoryExists(filePath) {
  const dirname = path.dirname(filePath);
  if (!fs.existsSync(dirname)) {
    fs.mkdirSync(dirname, { recursive: true });
  }
}

// POST Endpoint to initiate training
async function trainAndSaveModel(
  numPoints,
  numFeatures,
  numLabels,
  modelSavePath
) {
  try {
    // Ensure the directory exists
    ensureDirectoryExists(modelSavePath);

    // Generate dummy data
    const data = createDummyData(numPoints, numFeatures, numLabels);
    // Create a new model
    const model = createModel(numFeatures);
    // Train the model with dummy data
    await trainModel(model, data.features, data.labels);
    // Save the trained model to the specified path as JSON
    saveModelAsJSON(model, modelSavePath);
    console.log("Model trained and saved successfully");
  } catch (error) {
    console.error("Error during training and saving:", error.message);
  }
}

// Function to load the saved model and make predictions
async function loadAndPredict(modelSavePath, inputData) {
  try {
    // Load the saved model from JSON
    const model = await tf.loadLayersModel(`file://${modelSavePath}`);
    // Make predictions using the loaded model
    const prediction = await predict(model, inputData);
    // Determine the predicted class based on the prediction value
    const predictedClass = prediction.dataSync()[0] > 0.5 ? 1 : 0;
    console.log("Predicted class:", predictedClass);
  } catch (error) {
    console.error("Error during loading and predicting:", error.message);
  }
}

const numPoints = 100;
const numFeatures = 5;
const numLabels = 1;
const modelSavePath = "./saveModel/model.json";

// Train and save the model
trainAndSaveModel(numPoints, numFeatures, numLabels, modelSavePath);

// Load the saved model and make predictions
const inputData = tf.randomNormal([1, numFeatures]).arraySync();
loadAndPredict(modelSavePath, inputData);
