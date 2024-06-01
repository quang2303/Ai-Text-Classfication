// Import TensorFlow.js và các thư viện cần thiết
const toxicity = require("@tensorflow-models/toxicity");

// Tải mô hình toxicity đã được huấn luyện trước cho tiếng Việt
toxicity.load(0.8).then((model) => {
  // Hàm kiểm tra độc hại của văn bản
  function detectToxicity(text) {
    model.classify(text).then((predictions) => {
      // Xử lý dự đoán
      predictions.forEach((prediction) => {
        // Console log label
        console.log(prediction.label + ":");
        // Console log từng phần tử trong mảng kết quả
        prediction.results.forEach((result) => {
          console.log(result);
        });
      });
    });
  }

  // Sử dụng hàm detectToxicity để kiểm tra độc hại của văn bản
  const text = "Văn bản cần kiểm tra độc hại cái con mẹ mày fuck you";
  detectToxicity(text);
});
