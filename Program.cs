using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

class Program
{
  static void Main(string[] args)
  {
    // Step 1: Initialize MLContext
    MLContext mlContext = new MLContext();

    // Step 2: Load Data
    var data = LoadData(mlContext);

    // Step 3: Define the Processing Pipeline
    var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", nameof(NetworkTrafficData.PacketCount), nameof(NetworkTrafficData.AveragePacketSize))
      .Append(mlContext.Transforms.NormalizeMinMax("Features")); // Normalizing for better performance

    // Step 4: Define the Trainer
    var trainer = mlContext.BinaryClassification.Trainers.FastTree(
      labelColumnName: "Label",
      featureColumnName: "Features",
      numberOfLeaves: 50,
      learningRate: 0.05,
      numberOfTrees: 200);

    var trainingPipeline = dataProcessPipeline.Append(trainer);

    // Step 5: Train the Model
    var model = trainingPipeline.Fit(data);

    // Step 6: Evaluate the Model
    EvaluateModel(mlContext, model, data);

    // Step 7: Test Predictions
    var predictionEngine = mlContext.Model.CreatePredictionEngine<NetworkTrafficData, AnomalyPrediction>(model);

    var testData = new[]
    {
      new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 500 }, // Example of normal traffic
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500 }  // Example of potential anomaly
    };

    Console.WriteLine("Predictions:");
    foreach (var traffic in testData)
    {
      var prediction = predictionEngine.Predict(traffic);
      Console.WriteLine($"PacketCount: {traffic.PacketCount}, AveragePacketSize: {traffic.AveragePacketSize}");
      Console.WriteLine($"Prediction: {(prediction.Prediction ? "Normal" : "Anomaly")}, Score: {prediction.Score}, Probability: {prediction.Probability:P2}");
    }
  }

  // Load synthetic data for training
  static IDataView LoadData(MLContext mlContext)
  {
    var data = new[]
    {
      new NetworkTrafficData { PacketCount = 120, AveragePacketSize = 600, Label = true },
      new NetworkTrafficData { PacketCount = 200, AveragePacketSize = 800, Label = true },
      new NetworkTrafficData { PacketCount = 130, AveragePacketSize = 750, Label = true },
      new NetworkTrafficData { PacketCount = 110, AveragePacketSize = 680, Label = true },
      new NetworkTrafficData { PacketCount = 100, AveragePacketSize = 500, Label = true },
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500, Label = false }, // Anomalous traffic
      new NetworkTrafficData { PacketCount = 450, AveragePacketSize = 1600, Label = false },
      new NetworkTrafficData { PacketCount = 470, AveragePacketSize = 1700, Label = false },
      new NetworkTrafficData { PacketCount = 490, AveragePacketSize = 1800, Label = false }
    };

    return mlContext.Data.LoadFromEnumerable(data);
  }

  // Evaluate model accuracy
  static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data)
  {
    var predictions = model.Transform(data);
    var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine($"Model accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
  }
}
