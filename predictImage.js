const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
const fs = require("fs").promises;

const TARGET_CLASSES = {
  0: "Áo",
  1: "Bàn",
  2: "Board game",
  3: "Búp bê",
  4: "Chén",
  5: "Dây chuyền",
  6: "Đèn",
  7: "Đồ chơi trẻ em",
  8: "Động vật",
  9: "Đũa",
  10: "Ghế",
  11: "Giày & Dép",
  12: "Khuyên tai",
  13: "Ly",
  14: "Nhẫn",
  15: "Quần",
  16: "Thú nhồi bông",
  17: "Trang trí nội thất",
  18: "Trò chơi xếp hình",
  19: "Vòng tay",
  20: "Vũ khí",
};

let model;
let modelLoaded = false;

// Function to load the model
async function loadModel() {
  if (!modelLoaded) {
    model = await tf.loadGraphModel("file://./model_Image/model.json");
    modelLoaded = true;
  }
}

// Function to preprocess image buffer
function preprocessImage(imageBuffer) {
  return tf.tidy(() => {
    let tensor = tf.node
      .decodeImage(imageBuffer, 3)
      .resizeNearestNeighbor([224, 224]) // change the image size
      .expandDims()
      .toFloat()
      .reverse(-1); // RGB -> BGR
    return tensor;
  });
}

// Function to read image file and return buffer
async function readImage(filePath) {
  const imageBuffer = await fs.readFile(filePath);
  return imageBuffer;
}

// Function to make predictions on an array of image files
async function predictImages(imageFiles) {
  await loadModel();

  const predictions = [];

  for (const imageFile of imageFiles) {
    const imageBuffer = await readImage(imageFile);
    const preprocessedImage = preprocessImage(imageBuffer);

    const prediction = model.predict(preprocessedImage);
    const predictionArray = await prediction.data();

    // Find the index of the highest probability
    const maxProbIndex = prediction.argMax(-1).dataSync()[0];
    const maxProbValue = predictionArray[maxProbIndex];

    const predictedClass = TARGET_CLASSES[maxProbIndex];

    predictions.push({
      file: imageFile,
      className: predictedClass,
      probability: maxProbValue,
    });
  }

  console.log(predictions);

  return predictions;
}

module.exports = {
  predictImages,
};
