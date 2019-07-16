import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import * as Plotly from "plotly.js-dist";
import _ from "lodash";

Papa.parsePromise = function(file) {
  return new Promise(function(complete, error) {
    Papa.parse(file, {
      header: true,
      download: true,
      dynamicTyping: true,
      complete,
      error
    });
  });
};

const prepareData = async () => {
  const csv = await Papa.parsePromise(
    "https://raw.githubusercontent.com/curiousily/Customer-Churn-Detection-with-TensorFlow-js/master/src/data/customer-churn.csv"
  );

  const data = csv.data;
  return data.slice(0, data.length - 1);
};

const renderHistogram = (container, data, column, config) => {
  const defaulted = data
    .filter(r => r["default.payment.next.month"] === 1)
    .map(r => r[column]);
  const paid = data
    .filter(r => r["default.payment.next.month"] === 0)
    .map(r => r[column]);

  const dTrace = {
    name: "Defaulted",
    x: defaulted,
    type: "histogram",
    opacity: 0.35,
    marker: {
      color: "mediumvioletred"
    }
  };

  const hTrace = {
    name: "Paid",
    x: paid,
    type: "histogram",
    opacity: 0.35,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot(container, [dTrace, hTrace], {
    barmode: "overlay",
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

const renderDefaults = data => {
  const paymentStatus = data.map(r => r["default.payment.next.month"]);

  const [defaulted, paid] = _.partition(paymentStatus, o => o === 1);

  const chartData = [
    {
      labels: ["Defaulted", "Paid"],
      values: [defaulted.length, paid.length],
      type: "pie",
      opacity: 0.6,
      marker: {
        colors: ["mediumvioletred", "dodgerblue"]
      }
    }
  ];

  Plotly.newPlot("defaults-cont", chartData, {
    title: "Defaulted vs Paid payment"
  });
};

const renderSexDefault = data => {
  const defaulted = data.filter(r => r["default.payment.next.month"] === 1);
  const paied = data.filter(r => r["default.payment.next.month"] === 0);

  const [dMale, dFemale] = _.partition(defaulted, s => s.SEX === 1);
  const [pMale, pFemale] = _.partition(paied, b => b.SEX === 1);

  var sTrace = {
    x: ["Male", "Female"],
    y: [dMale.length, dFemale.length],
    name: "Defaulted",
    type: "bar",
    opacity: 0.6,
    marker: {
      color: "mediumvioletred"
    }
  };

  var bTrace = {
    x: ["Male", "Female"],
    y: [pMale.length, pFemale.length],
    name: "Paid",
    type: "bar",
    opacity: 0.6,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot("sex-default-cont", [sTrace, bTrace], {
    barmode: "group",
    title: "Sex vs Default Status"
  });
};

const renderEducationDefault = data => {
  const defaulted = data.filter(r => r["default.payment.next.month"] === 1);
  const paied = data.filter(r => r["default.payment.next.month"] === 0);

  const defaultedGroups = _.groupBy(defaulted, "EDUCATION");
  const paidGroups = _.groupBy(paied, "EDUCATION");

  var sTrace = {
    x: ["Graduate school", "University", "High school", "Other"],
    y: [
      defaultedGroups[1].length,
      defaultedGroups[2].length,
      defaultedGroups[3].length,
      defaultedGroups[4].length +
        defaultedGroups[5].length +
        defaultedGroups[6].length
    ],
    name: "Defaulted",
    type: "bar",
    opacity: 0.6,
    marker: {
      color: "mediumvioletred"
    }
  };

  var bTrace = {
    x: ["Graduate school", "University", "High school", "Other"],
    y: [
      paidGroups[1].length,
      paidGroups[2].length,
      paidGroups[3].length,
      paidGroups[0].length +
        paidGroups[4].length +
        paidGroups[5].length +
        paidGroups[6].length
    ],
    name: "Paid",
    type: "bar",
    opacity: 0.6,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot("edu-default-cont", [sTrace, bTrace], {
    barmode: "group",
    title: "Education vs Default Status"
  });
};

const VARIABLE_CATEGORY_COUNT = {
  MARRIAGE: 4,
  SEX: 2
};

// normalized = (value − min_value) / (max_value − min_value)
const normalize = tensor =>
  tf.div(
    tf.sub(tensor, tf.min(tensor)),
    tf.sub(tf.max(tensor), tf.min(tensor))
  );

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

const toTensors = (data, categoricalFeatures) => {
  // const features = Object.keys(data[0]).filter(
  //   f => f !== "default.payment.next.month" && f !== "ID"
  // );

  const categoricalData = {};
  categoricalFeatures.forEach(f => {
    categoricalData[f] = toCategorical(data, f);
  });

  const features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
  ].concat(Array.from(categoricalFeatures));

  const X = data.map((r, i) =>
    features.flatMap(f => {
      if (categoricalFeatures.has(f)) {
        return categoricalData[f][i];
      }

      return r[f];
    })
  );

  const y = tf.tensor(toCategorical(data, "Churn"));

  return [normalize(tf.tensor2d(X)), y];
};

const trainModel = async (xTrain, yTrain) => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      inputShape: [xTrain.shape[1]]
    })
  );

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu"
    })
  );

  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");

  await model.fit(xTrain, yTrain, {
    batchSize: 32,
    epochs: 30,
    shuffle: true,
    validationSplit: 0.1,
    callbacks: tfvis.show.fitCallbacks(
      lossContainer,
      ["loss", "val_loss", "acc", "val_acc"],
      {
        callbacks: ["onEpochEnd"]
      }
    )

    // callbacks: {
    //   onEpochEnd: async (epoch, logs) => {
    //     // console.log(logs);
    //     trainLogs.push(logs);
    //     tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
    //     tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
    //   }
    // }
  });

  return model;
};

const toCategorical = (data, column) => {
  const values = data.map(r => r[column]);
  const uniqueValues = new Set(values);

  const mapping = {};

  Array.from(uniqueValues).forEach((i, v) => {
    mapping[i] = v;
  });

  const encoded = values
    .map(v => {
      if (!v) {
        return 0;
      }
      return mapping[v];
    })
    .map(v => oneHot(v, uniqueValues.size));

  return encoded;
};

const run = async () => {
  const data = await prepareData();
  // toCategorical(data, "Churn");
  // console.log(data[data.length - 1]);

  // renderDefaults(data);
  // renderHistogram("limit-cont", data, "LIMIT_BAL", {
  //   title: "Amount of given credit",
  //   xLabel: "Limit"
  // });
  // renderHistogram("age-default-cont", data, "AGE", {
  //   title: "Age vs Payment Status",
  //   xLabel: "Age (years)"
  // });
  // renderSexDefault(data);
  // renderEducationDefault(data);
  // renderHistogram("repayment-cont", data, "PAY_0", {
  //   title: "Repayment status (September 2005)",
  //   xLabel: "Status"
  // });

  const [xTrain, yTrain] = toTensors(
    data,
    new Set([
      "TechSupport",
      "Contract",
      "PaymentMethod",
      "gender",
      "Partner",
      "InternetService",
      "Dependents",
      "PhoneService",
      "TechSupport",
      "StreamingTV",
      "PaperlessBilling"
    ])
  );

  const model = await trainModel(xTrain, yTrain);
  // const preds = model.predict(xTest).argMax(-1);
  // const labels = yTest.argMax(-1);
  // const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  // const container = document.getElementById("confusion-matrix");
  // tfvis.render.confusionMatrix(container, {
  //   values: confusionMatrix,
  //   tickLabels: ["Strike", "Ball"]
  // });
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
