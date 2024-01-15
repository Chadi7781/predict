
const tfModel = require("../model/tensorflowModel");
const tf = require("@tensorflow/tfjs-node");
// Logistic Regression Example
const logisticXs = tf.tensor1d([1, 2, 3, 4]);
const logisticYs = tf.tensor1d([0, 1, 1, 0]); // Binary labels (0 or 1)


function createLogisticModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 1, inputShape: [1], activation: "sigmoid" })
  );
  model.compile({
    optimizer: "sgd",
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

const logisticModel = createLogisticModel();

tfModel.trainModel(logisticModel, logisticXs, logisticYs, 100).then(() => {
  tfModel.predict(logisticModel, [5]);
});