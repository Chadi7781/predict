const tf = require("@tensorflow/tfjs-node");
const iris = require("iris-dataset");

// Load the iris dataset
const { data, labels } = iris.getData();
const features = tf.tensor2d(data);
const labelsTensor = tf.tensor2d(labels, [labels.length, labels[0].length]); // Assuming multiple output dimensions

// Normalize the features
const min = features.min(0);
const max = features.max(0);
const normalizedFeatures = features.sub(min).div(max.sub(min));

// Split the data into training and testing sets
const [trainFeatures, testFeatures] = tf.split(normalizedFeatures, 2);
const [trainLabels, testLabels] = tf.split(labelsTensor, 2);

// Implement k-Nearest Neighbors function for multiple inputs and outputs
function knn(predictFeatures, k = 3) {
  const { shape } = trainFeatures;

  // Calculate distances
  const distances = tf
    .sub(trainFeatures, predictFeatures)
    .square()
    .sum(1)
    .sqrt();

  // Get indices of k nearest neighbors
  const topKIndices = distances.argTopK(k).dataSync();

  // Get labels of the k nearest neighbors
  const topKLabels = topKIndices.map((index) =>
    trainLabels.dataSync().slice(index * shape[1], (index + 1) * shape[1])
  );

  // Calculate the mode (most common label) of the k nearest neighbors for each output dimension
  const mode = topKLabels[0].map((_, i) => {
    const columnValues = topKLabels.map((row) => row[i]);
    return tf.squeeze(tf.mode(columnValues).mode).arraySync()[0];
  });

  return mode;
}

// Test the kNN function
const testPredictions = testFeatures
  .arraySync()
  .map((features) => knn(tf.tensor2d([features]), 3));

console.log("Test Predictions:", testPredictions);
