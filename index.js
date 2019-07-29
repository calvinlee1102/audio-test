let recognizer;

function predictWord() {
    // Array of words that the recognizer is trained to recognize.
    const words = recognizer.wordLabels();
    recognizer.listen(({ scores }) => {
        // Turn scores into a list of (score,word) pairs.
        scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
        // Find the most probable word.
        scores.sort((s1, s2) => s2.score - s1.score);
        document.querySelector('#console').textContent = scores[0].word;
    }, { probabilityThreshold: 0.75 });
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    //predictWord();
    buildModel();
    //getVoiceModel();
}

app();

function collect(label) {
    if (recognizer.isListening()) {
        return recognizer.stopListening();
    }
    if (label == null) {
        return;
    }
    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({ vals, label });
        document.querySelector('#console').textContent =
            `${examples.length} examples collected`;
    }, {
            overlapFactor: 0.999,
            includeSpectrogram: true,
            invokeCallbackOnNoiseAndUnknown: true
        });
}

function normalize(x) {
    const mean = -100;
    const std = 10;
    return x.map(x => (x - mean) / std);
}

// One frame is ~23ms of audio.
const NUM_FRAMES = 30;
let examples = [];
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
var classes = 6;
let model;

async function train() {
    toggleButtons(false);
    const ys = tf.oneHot(examples.map(e => e.label), classes);
    const xsShape = [examples.length, ...INPUT_SHAPE];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

    await model.fit(xs, ys, {
        batchSize: 16,
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.querySelector('#console').textContent =
                    `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
            }
        }
    });
    tf.dispose([xs, ys]);
    toggleButtons(true);
}

async function buildModel() {
    const model2 = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/model.json');
    //model = tf.sequential();
    //model.add(tf.layers.depthwiseConv2d({
    //    depthMultiplier: 8,
    //    kernelSize: [NUM_FRAMES, 3],
    //    activation: 'relu',
    //    inputShape: INPUT_SHAPE
    //}));
    //model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
    //model.add(tf.layers.flatten());
    model = tf.sequential({layers: model2.layers.slice(0,12)});
    model.add(tf.layers.dense({ units: classes, activation: 'softmax' }));

    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}


function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));
    return result;
}

function getVoiceModel() {
    model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [NUM_FRAMES, 232, 1],
        kernelSize: [2, 8],
        filters: 8,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.conv2d({
        kernelSize: [2, 4],
        filters: 32,
        strides: [1, 1],
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [1, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dropout({
        rate: 0.25
    }));
    model.add(tf.layers.dense({
        units: 2000,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    model.add(tf.layers.dropout({
        rate: 0.5
    }));
    model.add(tf.layers.dense({
        units: 6,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}

async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('console').textContent = labeltext[label];
    if (label == 2) {
        return;
    }
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value =
        prevValue + (label === 0 ? -delta : delta);
}

function listen() {
    if (recognizer.isListening()) {
        recognizer.stopListening();
        toggleButtons(true);
        document.getElementById('listen').innerText = 'Listen';
        return;
    }
    toggleButtons(false);
    document.getElementById('listen').innerText = 'Stop';
    document.getElementById('listen').disabled = false;

    recognizer.listen(async ({ spectrogram: { frameSize, data } }) => {
        const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
        const probs = model.predict(input);
        const predLabel = probs.argMax(1);
        await moveSlider(predLabel);
        tf.dispose([input, probs, predLabel]);
    }, {
            overlapFactor: 0.999,
            includeSpectrogram: true,
            invokeCallbackOnNoiseAndUnknown: true,
            probabilityThreshold: 0.9
        });
}