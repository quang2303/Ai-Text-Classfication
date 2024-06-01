const tf = require("@tensorflow/tfjs");
const natural = require("natural");
const fs = require("fs");
require("@tensorflow/tfjs-node");
const dataTrain = require("./data/Negative");
require("@tensorflow/tfjs-node");

const MAX_SEQUENCE_LENGTH = 50;

const labels = dataTrain.map((item) => item.label);

// Chuyển đổi nhãn thành dạng số
const uniqueLabels = Array.from(new Set(labels));
const labelIndices = labels.map((label) => uniqueLabels.indexOf(label));

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

const textsLowerCase = dataTrain.map((item) => preprocess(item.text));

// Tạo từ điển từ vựng và ánh xạ từ vựng thành chỉ số
const vocabulary = new Set(textsLowerCase.flatMap((text) => text.split(" ")));

const wordIndex = {};
let index = 0;
vocabulary.forEach((word) => {
  wordIndex[word] = index;
  index++;
});

// Chuyển các chuỗi thành Tensor và làm đều chiều dài
const sequences = textsLowerCase.map((text) =>
  text
    .split(" ")
    .map((word) => wordIndex[word] || 0)
    .slice(0, MAX_SEQUENCE_LENGTH)
);
const paddedSequences = sequences.map((seq) => {
  const padLength = Math.max(0, MAX_SEQUENCE_LENGTH - seq.length);
  return seq.concat(Array(padLength).fill(0));
});
const paddedSequencesFloat32 = tf.tensor2d(paddedSequences);

// Chuyển đổi nhãn thành dạng tensor
const labelTensors = labelIndices.map((labelIndex) =>
  tf.oneHot(labelIndex, uniqueLabels.length).expandDims(0)
);
const labelTensorStacked = tf.concat(labelTensors, 0);

// Xây dựng mô hình
const model = tf.sequential();
model.add(
  tf.layers.embedding({
    inputDim: Object.keys(wordIndex).length,
    outputDim: 16,
    inputLength: paddedSequencesFloat32.shape[1],
  })
);
model.add(tf.layers.flatten());
model.add(
  tf.layers.dense({ units: uniqueLabels.length, activation: "softmax" })
);

// Biên dịch mô hình
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});

// Huấn luyện mô hình
const epochs = 150;
model
  .fit(paddedSequencesFloat32, labelTensorStacked, { epochs })
  .then((history) => {
    console.log("Training complete!");

    const textToPredict = "Tranh trang trí tường";

    predictSingleText(model, textToPredict, textsLowerCase);

    const wordIndexJSON = JSON.stringify(wordIndex);
    // Write JSON string to file
    fs.writeFile("wordIndex.json", wordIndexJSON, (err) => {
      if (err) {
        console.error("Error writing wordIndex to file:", err);
      } else {
        console.log("wordIndex has been written to wordIndex.json");
      }
    });
    // Lưu mô hình
    model
      .save("file://./saved_model")
      .then(() => console.log("Model saved successfully"))
      .catch((err) => console.error("Error saving model:", err));
  })
  .catch((err) => console.error("Training error:", err));

function predictSingleText(model, textToPredict, textsLowerCase) {
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
  // Dự đoán
  const prediction = model.predict(inputTensor);
  // In kết quả dự đoán
  console.log("Prediction for text:", textToPredict);
  console.log("Prediction:", prediction.arraySync()[0]);

  // Gán nhãn cho dự đoán
  const maxProbIndex = prediction.argMax(-1).dataSync()[0];
  const predictedLabel = uniqueLabels[maxProbIndex];
  console.log("Predicted label:", predictedLabel);
}
