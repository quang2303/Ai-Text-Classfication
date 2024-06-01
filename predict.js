const tf = require("@tensorflow/tfjs-node");
const natural = require("natural");
const fs = require("fs");

const MAX_SEQUENCE_LENGTH = 50;

let wordIndex;

const uniqueLabels = [
  "Dây chuyền",
  "Nhẫn",
  "Vòng tay",
  "Quần",
  "Áo",
  "Giày dép",
  "Khuyên tai",
  "Chén",
  "Ly",
  "Đũa",
  "Ghế",
  "Bàn",
  "Đèn",
  "Boardgame",
  "Trò chơi xếp hình",
  "Búp bê",
  "Thú nhồi bông",
  "Trang trí nội thất",
];

// Load the model
async function loadModel() {
  const model = await tf.loadLayersModel("file://./saved_model/model.json");
  return model;
}

// Tạo tokenizer và tiền xử lý văn bản
const tokenizer = new natural.WordTokenizer();
const preprocess = (text) => {
  const lowered = text.toLowerCase();
  const tokens = tokenizer.tokenize(lowered);
  const filteredTokens = tokens.filter(
    (token) => !natural.stopwords.some((stopword) => stopword === token)
  );
  return filteredTokens.join(" ");
};

// Hàm dự đoán cho một đoạn văn bản
async function predictSingleText(textToPredict) {
  const model = await loadModel();

  // Chuyển đổi đoạn văn bản thành chữ thường
  const lowerCaseText = preprocess(textToPredict);

  // Tokenize đoạn văn bản và giới hạn độ dài
  const sequence = lowerCaseText
    .split(" ")
    .map((word) => wordIndex[word] || 0)
    .slice(0, MAX_SEQUENCE_LENGTH);
  // Làm đều chiều dài
  console.log(sequence);
  const paddedSequence = sequence.concat(
    Array(MAX_SEQUENCE_LENGTH - sequence.length).fill(0)
  );
  // Chuyển đổi sang tensor và thêm một chiều
  const inputTensor = tf.tensor2d(paddedSequence, [1, MAX_SEQUENCE_LENGTH]);

  const prediction = model.predict(inputTensor);

  console.log("Prediction for text:", textToPredict);
  console.log("Prediction:", prediction.arraySync()[0]);

  const maxProbIndex = prediction.argMax(-1).dataSync()[0];
  const predictedLabel = uniqueLabels[maxProbIndex];
  console.log("Predicted label:", predictedLabel);
}

fs.readFile("wordIndex.json", (err, data) => {
  if (err) {
    console.error("Error reading wordIndex.json:", err);
    return;
  }

  // Parse JSON string to JavaScript object và gán cho biến wordIndex
  wordIndex = JSON.parse(data);
});

// Sử dụng hàm dự đoán cho một đoạn văn bản
const textToPredict = "Áo kaki form rộng";
predictSingleText(textToPredict);
