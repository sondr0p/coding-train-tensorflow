let x_vals = [];
let y_vals = [];

let a, b, c;

const learningRate = 0.5;
const optimizer = tf.train.adam(learningRate);

function setup() {
    createCanvas(400, 400);

    a = tf.variable(tf.scalar(random(-1, 1)));
    b = tf.variable(tf.scalar(random(-1, 1)));
    c = tf.variable(tf.scalar(random(-1, 1)));
}

function mousePressed() {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);

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
        let px = map(x_vals[i], -1, 1, 0, width);
        let py = map(y_vals[i], -1, 1, height, 0);
        point(px, py);
    }

    const curveX = [];
    for (let x = -1; x < 1; x += 0.05) {
        curveX.push(x);
    }

    const ys = tf.tidy(() => predict(curveX));
    // dataSync is tricky, do not use in prod
    let curveY = ys.dataSync();
    ys.dispose();

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);

    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width);
        let y = map(curveY[i], -1, 1, height, 0);
        vertex(x, y);
    }
    endShape();

    console.log(tf.memory().numTensors);
}

function predict(x) {
    const tf_x = tf.tensor1d(x);
    // y = ax^2 + bx + c
    const tf_y = tf_x.square().mul(a)
        .add(tf_x.mul(b))
        .add(c);

    return tf_y;
}

// pred = guesses from prediction, labels = actual y values
function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}