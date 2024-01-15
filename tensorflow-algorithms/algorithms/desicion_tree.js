const tf = require("@tensorflow/tfjs-node");

// Additional sample data for training the decision tree
const additionalTrainingData = [
  {
    dataPoints: 1200,
    dimensionality: 3,
    dataType: "Categorical",
    distribution: "Uniform",
    visualization: "Bar Chart",
  },
  {
    dataPoints: 8000,
    dimensionality: 8,
    dataType: "Numeric",
    distribution: "Skewed",
    visualization: "3D Surface Plot",
  },
  {
    dataPoints: 2000,
    dimensionality: 4,
    dataType: "Mixed",
    distribution: "Clustered",
    visualization: "Parallel Coordinates Plot",
  },
];

// Combine the original and additional training data
const trainingData = [
  {
    dataPoints: 800,
    dimensionality: 2,
    dataType: "Numeric",
    distribution: "Uniform",
    visualization: "Scatter Plot",
  },
  // Add more original training data...

  {
    dataPoints: 5000,
    dimensionality: 5,
    dataType: "Mixed",
    distribution: "Skewed",
    visualization: "Heatmap",
  },
  // Add more original training data...

  {
    dataPoints: 15000,
    dimensionality: 12,
    dataType: "Categorical",
    distribution: "Clustered",
    visualization: "Treemap",
  },
  // Add more original training data...

  ...additionalTrainingData,
];
// Features and target variable
const features = ["dataPoints", "dimensionality", "dataType", "distribution"];
const targetVariable = "visualization";

// Convert the target variable to numerical labels
const labelMap = {
  "Scatter Plot": 0,
  "Bar Chart": 1,
  Heatmap: 2,
  "3D Surface Plot": 3,
  "Parallel Coordinates Plot": 4,
  Treemap: 5,
};

const ys = tf.oneHot(
  trainingData.map((item) => labelMap[item[targetVariable]]),
  6
);

// Convert the training data to tensors
const xs = tf.tensor(
  trainingData.map((item) => features.map((feature) => item[feature]))
);

// Normalize the training data
const normalizedXs = tf.div(tf.sub(xs, xs.min()), tf.sub(xs.max(), xs.min()));

// Define and train a simple model using TensorFlow.js
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [4], units: 8, activation: "relu" }));
model.add(tf.layers.dense({ units: 6, activation: "softmax" }));

model.compile({
  optimizer: tf.train.adam({ learningRate: 0.01 }),
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

model.fit(normalizedXs, ys, { epochs: 500 }).then((history) => {
  console.log("Training History:", history);

  // Sample data for making predictions
  const newData = {
    dataPoints: 3000,
    dimensionality: 3,
    dataType: "Numeric",
    distribution: "Skewed",
  };

  // Convert the new data to a tensor
  const inputTensor = tf.tensor([features.map((feature) => newData[feature])]);

  // Normalize the new data
  const normalizedInputTensor = tf.div(
    tf.sub(inputTensor, xs.min()),
    tf.sub(xs.max(), xs.min())
  );

  // Make a prediction using the trained model
  const predictionTensor = model.predict(normalizedInputTensor);
  const predictedClass = tf.argMax(predictionTensor, (axis = 1)).dataSync()[0];

  // Convert the predicted class back to the original label
  const predictedLabel = Object.keys(labelMap).find(
    (key) => labelMap[key] === predictedClass
  );

  console.log(`Predicted Visualization: ${predictedLabel}`);
});
