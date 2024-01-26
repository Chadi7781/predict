const tf = require("@tensorflow/tfjs-node");

// Function to generate synthetic data for training
const generateData = (numPoints, coeff, sigma = 0.04) => {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d),
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    const three = tf.scalar(3, "int32");
    const ys = a
      .mul(xs.pow(three)) // ax^3
      .add(b.mul(xs.square())) // + bx^2
      .add(c.mul(xs)) // + cx
      .add(d) // + d
      .add(tf.randomNormal([numPoints], 0, sigma)); // Some noise

    // Normalize the y values to the range 0 to 1.
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized,
    };
  });
};

// Function to train the model
async function trainModel(xs, ys) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.add(tf.layers.dense({ units: 4, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));

  model.compile({
    loss: "meanSquaredError",
    optimizer: "sgd",
  });

  await model.fit(xs, ys, {
    epochs: 100,
  });

  return model;
}

// Function to run training and prediction
async function run() {
  // Use static values for coefficients and number of data points
  const staticCoefficients = {
    a: 0.5,
    b: -0.2,
    c: 0.8,
    d: 0.3,
  };
  const numDataPoints = 100;

  // Generate training data based on static coefficients
  const trainingData = generateData(numDataPoints, staticCoefficients);

  // Prepare the inputs and labels for training
  const xs = trainingData.xs;
  const ys = trainingData.ys;

  // Train the model
  const trainedModel = await trainModel(xs, ys);

  // Predict output for a new set of data points
  const testXs = tf.linspace(-1, 1, 100);
  const preds = trainedModel.predict(testXs.reshape([100, 1]));

  // Un-normalize the predictions
  const unNormPreds = preds.mul(ys.max().sub(ys.min())).add(ys.min());

  // Output the predictions
  const predictionsArray = await unNormPreds.array();
  console.log("Output predictions:", predictionsArray);

  // Cleanup memory
  trainedModel.dispose();
  unNormPreds.dispose();
  testXs.dispose();
}

// Run the script
run();
