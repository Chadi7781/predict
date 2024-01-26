const tf = require("@tensorflow/tfjs-node");

class KMeans {
  constructor(k, iterations) {
    this.k = k;
    this.iterations = iterations;
  }

  fit(data) {
    this.centroids = data.slice(0, this.k);
    for (let i = 0; i < this.iterations; i++) {
      const clusters = this.createClusters(data);
      this.centroids = this.calculateCentroids(clusters);
    }
  }

  predict(data) {
    return data.map((point) => this.predictPoint(point));
  }

  createClusters(data) {
    return data.reduce(
      (clusters, point) => {
        const centroidIndex = this.findClosestCentroid(point);
        clusters[centroidIndex].push(point);
        return clusters;
      },
      Array.from({ length: this.k }, () => [])
    );
  }

  calculateCentroids(clusters) {
    return clusters.map((cluster) => {
      const sum = cluster.reduce(
        (sum, point) => tf.add(sum, point),
        tf.zeros([1])
      );
      return tf.div(sum, cluster.length).arraySync();
    });
  }

  findClosestCentroid(point) {
    const distances = this.centroids.map((centroid) =>
      tf.norm(tf.sub(point, centroid)).arraySync()
    );
    return distances.indexOf(Math.min(...distances));
  }

  predictPoint(point) {
    return this.findClosestCentroid(point);
  }
}

// Generate a dataset of random numbers
const data = Array.from({ length: 100 }, () => [Math.random()]);

const kmeans = new KMeans(6, 100);
kmeans.fit(data);
const predictions = kmeans.predict(data);
console.log(predictions);

end

