// Create a manual dataset
const data = [
  [1.0, 2.0],
  [2.0, 3.0],
  [3.0, 4.0],
  [4.0, 5.0],
  // Add more data as needed
];

const labels = [
  [0],
  [0],
  [1],
  [1],
  // Add more labels as needed
];

// Convert the dataset to JavaScript arrays
const features = data.map((row) => row.map((val) => val));
const labelsArray = labels.map((row) => row[0]);

// Normalize the features
const min = Math.min(...features.flat());
const max = Math.max(...features.flat());
const normalizedFeatures = features.map((row) =>
  row.map((val) => (val - min) / (max - min))
);

// Split the data into training and testing sets
const splitIndex = Math.floor(normalizedFeatures.length / 2);
const trainFeatures = normalizedFeatures.slice(0, splitIndex);
const testFeatures = normalizedFeatures.slice(splitIndex);
const trainLabels = labelsArray.slice(0, splitIndex);
const testLabels = labelsArray.slice(splitIndex);

// Implement k-Nearest Neighbors function for multiple inputs and outputs
function knn(predictFeatures, k = 3) {
  // Calculate distances
  const distances = trainFeatures.map((trainRow) =>
    Math.sqrt(
      trainRow.reduce((sum, val, i) => sum + (val - predictFeatures[i]) ** 2, 0)
    )
  );

  // Get indices of k nearest neighbors
  const topKIndices = distances
    .map((_, i) => i)
    .sort((a, b) => distances[a] - distances[b])
    .slice(0, k);

  // Get labels of the k nearest neighbors
  const topKLabels = topKIndices.map((index) => trainLabels[index]);

  console.log("Distances:", distances);
  console.log("Top k Indices:", topKIndices);
  console.log("Top k Labels:", topKLabels);

  // Calculate the mode (most common label)
  const mode = topKLabels.reduce((acc, label) => {
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});

  console.log("Mode:", mode);

  return Object.keys(mode).reduce((a, b) => (mode[a] > mode[b] ? a : b));
}

// Test the kNN function
const testPredictions = testFeatures.map((features) => knn(features, 3));

console.log("Test Predictions:", testPredictions);
