const readline = require("readline");
const fs = require("fs");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Get a list of available files in the current directory
const files = fs
  .readdirSync("./algorithms")
  .filter((file) => file.endsWith(".js"));

// Display the list of available files
console.log("Available files:");
files.forEach((file, index) => {
  console.log(`${index + 1}. ${file}`);
});

rl.question("Enter the name of the file you want to execute: ", (fileName) => {
  const filePath = `./algorithms/${fileName}`;

  // Check if the selected file exists
  if (fs.existsSync(filePath)) {
    // If the file exists, import and execute  it
    const importedModule = require(filePath);
    console.log(`Algorithm ${fileName} executed.`);
  } else {
    console.log(`File ${fileName} not found.`);
  }

  // Close the readline interface
  rl.close();
});
