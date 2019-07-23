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
    }, { probabilityThreshold: 0.95 });
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    //predictWord();
}

app();

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

function collect(label) {
    if (recognizer.isListening()) {
        alert('over');
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

function getVoiceModel() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        inputShape: [43, 232, 1],
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
        units: 20,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));
    //var optimizer = tf.train.sgd(LEARNING_RATE);
    //alert(optimizertype);
    //if (optimizertype =='sgd') {
    //    optimizer = tf.train.sgd(LEARNING_RATE);
    //} else if (optimizertype == 'momentum') {
    //    optimizer = tf.train.momentum(LEARNING_RATE);
    //}
    //model.compile({
    //    optimizer: optimizer,
    //    loss: 'categoricalCrossentropy',
    //    metrics: ['accuracy']
    //});
    return model;
}