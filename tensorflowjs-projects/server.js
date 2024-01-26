const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const app = express();
const port = 3000;

app.use(express.static("public")); // Serve static files from the 'public' directory

app.use("/utils", express.static("utils"));

const routerLinearRegression = require("./projects/LinearRegression/linear-regression");
const routerLogisticRegression = require("./projects/LogisiticRegClassification/logisitic-regression");

app.use("/project1", routerLinearRegression);
app.use("/project2", routerLogisticRegression);

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
