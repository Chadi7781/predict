const tf = require("@tensorflow/tfjs-node");
const csv = require("csv-parser");
const fs = require("fs");

class KNN {
  constructor(trainingSet, labels, k) {
    this.trainingSet = tf.tensor2d(trainingSet);
    this.labels = tf.tensor1d(labels);
    this.k = k;
  }

  classify(newExample) {
    const newExampleTensor = tf.tensor1d(newExample);
    const distances = this.trainingSet
      .sub(newExampleTensor)
      .pow(2)
      .sum(1)
      .sqrt();
    const indices = tf.topk(distances.neg(), this.k).indices;
    const kNearestNeighbors = tf.gather(this.labels, indices);
    const counts = tf.bincount(kNearestNeighbors.toInt());
    const prediction = tf.argMax(counts);
    return prediction.arraySync();
  }

  calculateAccuracy(testSet, testLabels) {
    const testSetTensor = tf.tensor2d(testSet);
    const testLabelsTensor = tf.tensor1d(testLabels);
    const predictions = testSetTensor
      .arraySync()
      .map((example) => this.classify(example));
    const correctPredictions = predictions.filter(
      (prediction, i) => prediction === testLabels[i]
    ).length;
    const accuracy = correctPredictions / testSet.length;
    return accuracy;
  }
}

// Load Iris dataset
const data = [];
fs.createReadStream("../../utils/iris.csv")
  .pipe(csv())
  .on("data", (row) => {
    const features = Object.values(row).slice(0, 4).map(Number);
    const label =
      Object.values(row)[4] === "setosa"
        ? 0
        : Object.values(row)[4] === "versicolor"
        ? 1
        : 2;
    data.push({ features, label });
  })
  .on("end", () => {
    // Shuffle data
    tf.util.shuffle(data);

    // Split data into training set and test set
    const trainingSet = data.slice(0, 120).map((d) => d.features);
    const trainingLabels = data.slice(0, 120).map((d) => d.label);
    const testSet = data.slice(120).map((d) => d.features);
    const testLabels = data.slice(120).map((d) => d.label);

    // Create a new KNN classifier
    const knn = new KNN(trainingSet, trainingLabels, 3);

    // Calculate the accuracy
    const accuracy = knn.calculateAccuracy(testSet, testLabels);
    console.log(`Accuracy: ${accuracy}`);
  });
