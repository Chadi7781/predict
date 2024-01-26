const express = require("express");
const tf = require("@tensorflow/tfjs-node");

const app = express.Router();
const port = 3000;

// Normalize the input data
const normalize = (data) => {
  const max = Math.max(...data);
  const min = Math.min(...data);
  return data.map((value) => (value - min) / (max - min));
};

var x = normalize([
  3.5, 3, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3, 3, 4, 4.4, 3.9,
  3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1,
  4.2, 3.1, 3.2, 3.5, 3.1, 3, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3, 3.8, 3.2, 3.7,
  3.3, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2, 3, 2.2, 2.9, 2.9,
  3.1, 3, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3, 2.8, 3, 2.9, 2.6, 2.4, 2.4,
  2.7, 2.7, 3, 3.4, 3.1, 2.3, 3, 2.5, 2.6, 3, 2.6, 2.3, 2.7, 3, 2.9, 2.9, 2.5,
  2.8, 3.3, 2.7, 3, 2.9, 3, 3, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3, 2.5, 2.8, 3.2,
  3, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3, 2.8, 3, 2.8, 3.8, 2.8,
  2.8, 2.6, 3, 3.4, 3.1, 3, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3, 2.5, 3, 3.4, 3,
]);

var y = normalize([
  0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2,
  0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
  0.2, 0.4, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4,
  0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1, 1.3, 1.4, 1,
  1.5, 1, 1.4, 1.3, 1.4, 1.5, 1, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4,
  1.7, 1.5, 1, 1.1, 1, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1,
  1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8,
  2.5, 2, 1.9, 2.1, 2, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2, 2, 1.8, 2.1, 1.8,
  1.8, 1.8, 2.1, 1.6, 1.9, 2, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3,
  1.9, 2.3, 2.5, 2.3, 1.9, 2, 2.3, 1.8,
]);

var labels = normalize([
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]);

var points = [];
var iris = [];
var pred = [];
for (i = 0; i < labels.length; i++) {
  iris.push([x[i], y[i]]);
}
let accuracy;
let testAccuracy;
// Serve the index.html file
app.get("/", (req, res) => {
  res.sendFile(__dirname + "/index.html");
});

// Create the model
const model = tf.sequential();

// One fully connected layer
model.add(tf.layers.dense({ units: 1, activation: "sigmoid", inputDim: 2 }));

model.compile({
  optimizer: tf.train.sgd(0.01), // Adjust the learning rate here
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
});

// Handle predictions
app.get("/predict", (req, res) => {
  const swidth = parseFloat(req.query.swidth);
  const pwidth = parseFloat(req.query.pwidth);

  const bestfit = model.predict(tf.tensor([[swidth, pwidth]])).dataSync()[0];

  const result = {
    prediction: bestfit,
    point: { x: swidth, y: pwidth },
    color: bestfit <= 0.5 ? "#FF0000" : "#00FF00",
    accuracy: accuracy,
    testAccuracy: testAccuracy,
  };

  res.json(result);
});
app.get("/train", async (req, res) => {
  // Fitting the data
  const batchSize = 32;
  const epochs = 100;

  const trainData = tf.tensor(iris.slice(0, Math.floor(0.8 * labels.length)));
  const trainLabels = tf.tensor(
    labels.slice(0, Math.floor(0.8 * labels.length))
  );

  const testData = tf.tensor(iris.slice(Math.floor(0.8 * labels.length)));
  const testLabels = tf.tensor(labels.slice(Math.floor(0.8 * labels.length)));

  model
    .fit(trainData, trainLabels, {
      batchSize: batchSize,
      epochs: epochs,
      validationData: [testData, testLabels],
    })
    .then((info) => {
      console.log("Model finished training!");
      console.log(info.history.acc); // Training accuracy values
      console.log(info.history.val_acc); // Validation accuracy values

      accuracy = info.history.acc[info.history.acc.length - 1];
      testAccuracy = info.history.val_acc[info.history.val_acc.length - 1];

      console.log(`Final training accuracy: ${accuracy}`);
      console.log(`Final test accuracy: ${testAccuracy}`);

      const result = {
        accuracy,
      };

      res.json(result);
    });
});
module.exports = app;
