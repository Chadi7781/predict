const tf = require("@tensorflow/tfjs-node");

// Sequential model
const model = tf.sequential();

// Configure the LSTM layer
// Note: inputShape is based on the sequence length and features of your dataset
model.add(
  tf.layers.lstm({
    units: 50, // The number of LSTM units
    inputShape: [10, 1], // 10 timesteps and 1 feature per step
    returnSequences: false, // Set true if next layer is recurrent; false otherwise
  })
);

// Add a Dense output layer
model.add(
  tf.layers.dense({
    units: 1, // For a single continuous value as output
    activation: "linear",
  })
);

// Compile the model with a loss function and an optimizer
model.compile({
  loss: "meanSquaredError",
  optimizer: "adam",
});

// Create some dummy sequential data
const xs = tf.randomNormal([100, 10, 1]); // 100 sequences, each of length 10 with 1 feature
const ys = tf.randomNormal([100, 1]); // Each sequence has one output

// Train the model
async function trainModel() {
  await model.fit(xs, ys, {
    epochs: 50, // The number of iterations to train the data
    batchSize: 32, // The size of the batch by which the data will be processed
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}`);
      },
    },
  });

  // After training, the model can be used to predict on new data
  model.predict(tf.randomNormal([1, 10, 1])).print();
}

trainModel()
  .then(() => console.log("Model training complete"))
  .catch((err) => console.error("Model training encountered an error:", err));
