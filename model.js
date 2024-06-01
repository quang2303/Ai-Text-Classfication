const tf = require("@tensorflow/tfjs");
const natural = require("natural");
const dataTrain = require("./data/Negative");

// Tạo one-hot encoding cho các nhãn từ dữ liệu
const labels = Array.from(new Set(dataTrain.map((item) => item.label)));
const labelIndex = labels.reduce((acc, label, index) => {
  acc[label] = index;
  return acc;
}, {});

const oneHotEncode = (label) => {
  const index = labelIndex[label];
  const oneHot = new Array(labels.length).fill(0);
  oneHot[index] = 1;
  return oneHot;
};

// Ánh xạ văn bản vào one-hot encoding và xáo trộn dữ liệu
const shuffledDataTrain = dataTrain
  .map((item) => {
    const text = item.text;
    const label = oneHotEncode(item.label);
    return { text, label };
  })
  .sort(() => Math.random() - 0.5);

// Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
const data = shuffledDataTrain.map((item) => item.text);

const label = shuffledDataTrain.map((item) => item.label);

const trainSize = Math.floor(shuffledDataTrain.length * 0.8);

// Chia dữ liệu thành tập huấn luyện và tập kiểm tra
const trainData = data.slice(0, trainSize);
const testData = data.slice(trainSize);

// Tạo nhãn cho tập huấn luyện và tập kiểm tra
const trainLabels = label.slice(0, trainSize);
const testLabels = label.slice(trainSize);

const tokenizer = new natural.WordTokenizer();
const preprocess = (text) => {
  const lowered = text.toLowerCase();
  const tokens = tokenizer.tokenize(lowered);
  const filtered = tokens.filter(
    (token) => !natural.stopwords.some((stopword) => stopword === token)
  );
  return filtered.join(" ");
};
const trainDataNormalized = trainData.map(preprocess);
const testDataNormalized = testData.map(preprocess);

// Tạo từ điển từ vựng và ánh xạ từ vựng thành chỉ số
const vocabulary = new Set(
  trainDataNormalized.flatMap((textArray) => textArray.split(" "))
);

const wordIndex = {};
let index = 1; // Bắt đầu từ 1 vì 0 thường được sử dụng cho các từ không xác định
vocabulary.forEach((word) => {
  wordIndex[word] = index;
  index++;
});

const textToIndices = (texts) => {
  return texts.map((text) => {
    const tokens = preprocess(text).split(" ");
    return tokens.map((token) => wordIndex[token] || 0);
  });
};

const trainDataIndices = textToIndices(trainDataNormalized);
const testDataIndices = textToIndices(testDataNormalized);

// Xây dựng mô hình
const model = tf.sequential();
model.add(
  tf.layers.embedding({
    inputDim: 1000,
    outputDim: 32,
    inputLength: 100,
  })
);
model.add(
  tf.layers.bidirectional({
    layer: tf.layers.lstm({
      units: 32,
      returnSequences: true,
    }),
  })
);
model.add(tf.layers.globalAveragePooling1d());
model.add(
  tf.layers.dense({
    units: 6, // Số lượng lớp đích (số loại sản phẩm)
    activation: "softmax",
  })
);
model.compile({
  loss: "categoricalCrossentropy",
  optimizer: "adam",
  metrics: ["accuracy"],
});

const padSequences = (sequences, maxLen, padding = "PAD") => {
  return sequences.map((seq) => {
    if (seq.length >= maxLen) {
      return seq.slice(0, maxLen);
    } else {
      return [...seq, ...Array(maxLen - seq.length).fill(padding)];
    }
  });
};

// Padding cho dữ liệu huấn luyện và kiểm tra
const maxSeqLength = 100;
const trainDataPadded = padSequences(trainDataIndices, maxSeqLength);
const testDataPadded = padSequences(testDataIndices, maxSeqLength);

// Huấn luyện mô hình
model
  .fit(tf.tensor2d(trainDataPadded), tf.tensor2d(trainLabels), {
    epochs: 10,
    validationData: [tf.tensor2d(testDataPadded), tf.tensor2d(testLabels)],
  })
  .then(() => {
    // Đánh giá mô hình
    const results = model.evaluate(
      tf.tensor2d(testDataPadded),
      tf.tensor2d(testLabels)
    );
    console.log(results);

    // Dữ liệu đầu vào cho dự đoán
    const inputText = ["Quần dài"];

    // Chuyển đổi văn bản thành chỉ số
    const inputIndices = padSequences(textToIndices(inputText), 100);

    // Chuyển đổi dữ liệu đầu vào thành tensor
    const inputTensor = tf.tensor2d(inputIndices);

    // Dự đoán loại sản phẩm
    const prediction = model.predict(inputTensor);

    // Lấy giá trị dự đoán từ tensor
    const predictionValues = prediction.dataSync();

    // In ra tỉ lệ dự đoán
    console.log("Tỉ lệ dự đoán cho mỗi nhãn:");
    labels.forEach((label, index) => {
      console.log(`${label}: ${predictionValues[index]}`);
    });

    saveModel();
  });

async function saveModel() {
  try {
    console.log("Đang lưu mô hình.");
    await model.save(`./saved_model`);
    console.log("Mô hình đã được lưu thành công.");
  } catch (error) {
    console.error("Đã xảy ra lỗi khi lưu mô hình:", error);
  }
}
