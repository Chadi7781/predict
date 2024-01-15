const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const app = express();
const port = 3000;

app.use(express.static("public")); // Serve static files from the 'public' directory

app.use("/utils", express.static("utils"));

app.get("/read-csv", async (req, res) => {
  try {
    const parsedData = await parseCSV();
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

  const samplesDataset = houseDataSet.take(3);
  const dataArray = await samplesDataset.toArray();

  const points = houseDataSet.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  return await points.toArray();
}

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
