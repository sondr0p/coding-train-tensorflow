let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(400, 400);

    // need to be able to change, so use variable
    // generate random from 0 - 1 (p5 function)
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);

    x_vals.push(x);
    y_vals.push(y);
}

function draw() {
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const labels = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), labels));
        }
    })


    background(0);

    stroke(255);
    strokeWeight(8);
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], 0, 1, 0, width);
        let py = map(y_vals[i], 0, 1, height, 0);
        point(px, py);
    }

    const lineX = [0, 1];

    const ys = tf.tidy(() => predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();

    let x1 = map(lineX[0], 0, 1, 0, width);
    let x2 = map(lineX[1], 0, 1, 0, width);

    let y1 = map(lineY[0], 0, 1, height, 0);
    let y2 = map(lineY[1], 0, 1, height, 0);

    strokeWeight(2);
    line(x1, y1, x2, y2);
    console.log(tf.memory().numTensors);
}

function predict(x) {
    const tf_x = tf.tensor1d(x);
    // y = mx + b
    const tf_y = tf_x.mul(m).add(b);

    return tf_y;
}

// pred = guesses from prediction, labels = actual y values
function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}