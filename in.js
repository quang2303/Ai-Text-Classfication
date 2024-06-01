const { predictImages } = require("./predictImage.js");

const imageFiles = [
  "./Image/den.jpg",
  "./Image/ghe.jpg",
  "./Image/ly.jpg",
  // Add more image paths as needed
];

console.log(imageFiles)(async () => {
  try {
    const results = await predictImages(imageFiles);
    results.forEach((result) => {
      console.log(`Predictions for ${result.file}:`);
      console.log(`${result.className}: ${result.probability.toFixed(6)}`);
    });
  } catch (error) {
    console.error("Error making predictions:", error);
  }
})();
