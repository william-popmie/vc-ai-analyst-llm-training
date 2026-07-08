const path = require('path');
const ort = require('onnxruntime-node');

async function test() {
  const session = await ort.InferenceSession.create(path.join(__dirname, '../models/model.onnx'));
  const x = new Float32Array([5, 4, 5, 3, 5, 4, 220000]);
  const tensor = new ort.Tensor('float32', x, [1, 7]);
  
  const feeds = { float_input: tensor };
  const results = await session.run(feeds);
  
  const label = results.label.data[0];
  const investProb = results.probabilities.data[1];
  
  console.log(`Label: ${label}, Invest prob: ${(investProb * 100).toFixed(1)}%`);
}

test();
