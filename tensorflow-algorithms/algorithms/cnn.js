const tf = require("@tensorflow/tfjs-node");

// Define a sequential model
const model = tf.sequential();

// Add a convolutional layer
model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 32,
    activation: "relu",
    padding: "same",
  })
);

// Add a max pooling layer
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

// Add a dropout layer
model.add(tf.layers.dropout({ rate: 0.25 }));

// Add another convolutional layer
model.add(
  tf.layers.conv2d({
    kernelSize: 3,
    filters: 64,
    activation: "relu",
    padding: "same",
  })
);

// Add another max pooling layer
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

// Add another dropout layer
model.add(tf.layers.dropout({ rate: 0.25 }));

// Flatten the output of the convolutions
model.add(tf.layers.flatten());

// Add a dense layer
model.add(
  tf.layers.dense({
    units: 128,
    activation: "relu",
  })
);

// Add another dropout layer
model.add(tf.layers.dropout({ rate: 0.5 }));

// Output layer with softmax activation
model.add(
  tf.layers.dense({
    units: 10,
    activation: "softmax",
  })
);

// Compile the model with a different optimizer and loss function
model.compile({
  optimizer: "sgd",
  loss: "meanSquaredError",
  metrics: ["accuracy"],
});

// Dummy dataset example
const xs = tf.randomNormal([100, 28, 28, 1]);
const ys = tf.randomUniform([100, 10]).toInt();

// Train the model using the dummy data
async function train() {
  await model.fit(xs, ys, {
    batchSize: 32,
    epochs: 10,
    shuffle: true,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`
        ),
    },
  });
}

train();
