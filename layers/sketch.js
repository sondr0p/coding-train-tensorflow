

const model = tf.sequential();

// Create hidden layer
// dense is a "fully connected" layer
const hidden = tf.layers.dense({
    units: 4, // number of nodes
    inputShape: [2], // number of inputs
    activation: 'sigmoid'
});

// Create output layer
const output = tf.layers.dense({
    units: 1,
    // input shape is inferred from previous layer
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

// optimizer using gradient descent
const config = {
    optimizer: tf.train.sgd(0.1),
    loss: 'meanSquaredError'
}

// I'm done, now compile the model
model.compile(config);


const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1]
]);

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
]);

train().then(() => {
    console.log("training complete");
    let outputs = model.predict(xs);
    outputs.print();
});

async function train() {
    for (let i = 0; i < 10000; i++) {
        const response = await model.fit(xs, ys);
        console.log(response.history.loss[0]);
    }
}



