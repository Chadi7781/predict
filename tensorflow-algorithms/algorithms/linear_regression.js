const tf = require("@tensorflow/tfjs-node");

// Multiple inputs and outputs data
const xs1 = tf.tensor2d([[1], [2], [3], [4]]);
const xs2 = tf.tensor2d([[2], [3], [4], [5]]);
const ys1 = tf.tensor2d([[2], [4], [5], [4]]);
const ys2 = tf.tensor2d([[3], [5], [6], [5]]);

function createMultipleInputOutputModel() {
  // Define input shapes based on the number of features in each input
  const input1 = tf.input({ shape: [1] });
  const input2 = tf.input({ shape: [1] });

  // Combine inputs
  const mergedInputs = tf.layers.concatenate().apply([input1, input2]);

  // Add dense layer with the merged input
  const denseLayer = tf.layers.dense({ units: 1 }).apply(mergedInputs);

  // Split the output into multiple branches
  const output1 = tf.layers.dense({ units: 1 }).apply(denseLayer);
  const output2 = tf.layers.dense({ units: 1 }).apply(denseLayer);

  // Create the model
  const model = tf.model({
    inputs: [input1, input2],
    outputs: [output1, output2],
  });

  // Compile the model
  model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

  return model;
}

const multiInputOutputModel = createMultipleInputOutputModel();

// Train the model with multiple inputs and outputs
trainModel(
  multiInputOutputModel,
  [xs1, xs2], // Pass an array of input tensors
  [ys1, ys2], // Pass an array of output tensors
  100
).then(() => {
  // Predict with multiple inputs
  const predictions = predict(multiInputOutputModel, [
    tf.tensor2d([[5]]),
    tf.tensor2d([[6]]),
  ]);

  // Output predictions
  predictions.forEach((prediction, i) => {
    console.log(`Output ${i + 1}: ${prediction.dataSync()[0]}`);
  });
});

async function trainModel(model, inputTensors, outputTensors, epochs) {
  // Use the provided model, inputTensors, outputTensors, and epochs to train the model
  await model.fit(inputTensors, outputTensors, { epochs });
}

function predict(model, inputTensors) {
  // Use the provided model and inputTensors to make predictions
  return model.predict(inputTensors);
}
