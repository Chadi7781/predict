const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const app = express();
const port = 3000;

app.use(express.static("public")); // Serve static files from the 'public' directory

app.use("/utils", express.static("utils"));

app.get("/read-csv", async (req, res) => {
  try {
    const parsedData = await parseCSV(); // Pass 3 as the second argument to take only the first 3 lines
    res.json(parsedData);
  } catch (error) {
    console.error("Error reading CSV:", error.message);
    res.status(500).json({ error: "Internal server error" });
  }
});

async function parseCSV() {
  const houseDataSet = tf.data.csv(
    "http://localhost:3000/utils/kc_house_data.csv"
  );

  const samplesDataset = houseDataSet.take(1500);
  const dataArray = await samplesDataset.toArray();

  const points = houseDataSet.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  console.log(await points.toArray());

  //inputs
  const featureValues = await points.map((p) => p.x).toArray();
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  // outputs
  const labelValues = await points.map((p) => p.y).toArray();
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  featureTensor.print();
  labelTensor.print();

  return await points.toArray();
}

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
