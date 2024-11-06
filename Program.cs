using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

class Program
{
  static void Main(string[] args)
  {
    // Initialize MLContext
    MLContext mlContext = new MLContext();

    // Load Data
    var data = LoadData(mlContext);

    // Processing Pipeline
    var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", nameof(NetworkTrafficData.PacketCount), nameof(NetworkTrafficData.AveragePacketSize))
      .Append(mlContext.Transforms.NormalizeMinMax("Features"));

    // Trainer
    var trainer = mlContext.BinaryClassification.Trainers.FastTree(
    labelColumnName: "Label",
    featureColumnName: "Features",
    numberOfLeaves: 15,       // Moderate complexity
    learningRate: 0.03,       // Low learning rate for gradual learning
    numberOfTrees: 100        // Sufficiently large ensemble size
    );

    // Train and evaluate model
    var trainingPipeline = dataProcessPipeline.Append(trainer);
    var model = trainingPipeline.Fit(data);
    EvaluateModel(mlContext, model, data);

    // Prediction with a custom threshold
    float anomalyThreshold = 0.6f; // Example threshold
    var predictionEngine = mlContext.Model.CreatePredictionEngine<NetworkTrafficData, AnomalyPrediction>(model);
    var testData = new[]
    {
      new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 500 },  // normal traffic
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500 }  // anomaly
    };

    Console.WriteLine("Predictions:");
    foreach (var traffic in testData)
    {
      var prediction = predictionEngine.Predict(traffic);
      string result = prediction.Probability < anomalyThreshold ? "Anomaly" : "Normal";
      Console.WriteLine($"PacketCount: {traffic.PacketCount}, AveragePacketSize: {traffic.AveragePacketSize}");
      Console.WriteLine($"Prediction: {result}, Score: {prediction.Score}, Probability: {prediction.Probability:P2}");
    }
  }

  // synthetic data for training
  // Load more synthetic data for training
  static IDataView LoadData(MLContext mlContext)
  {
    var data = new[]
    {
      // Normal traffic samples
      new NetworkTrafficData { PacketCount = 100, AveragePacketSize = 500, Label = true },
      new NetworkTrafficData { PacketCount = 120, AveragePacketSize = 600, Label = true },
      new NetworkTrafficData { PacketCount = 130, AveragePacketSize = 750, Label = true },
      new NetworkTrafficData { PacketCount = 110, AveragePacketSize = 680, Label = true },
      new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 520, Label = true },
      new NetworkTrafficData { PacketCount = 160, AveragePacketSize = 700, Label = true },
      new NetworkTrafficData { PacketCount = 140, AveragePacketSize = 800, Label = true },
      new NetworkTrafficData { PacketCount = 180, AveragePacketSize = 900, Label = true },
      new NetworkTrafficData { PacketCount = 200, AveragePacketSize = 1000, Label = true },
      new NetworkTrafficData { PacketCount = 220, AveragePacketSize = 1050, Label = true },
      
      // Anomalous traffic samples
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500, Label = false },
      new NetworkTrafficData { PacketCount = 450, AveragePacketSize = 1600, Label = false },
      new NetworkTrafficData { PacketCount = 470, AveragePacketSize = 1700, Label = false },
      new NetworkTrafficData { PacketCount = 490, AveragePacketSize = 1800, Label = false },
      new NetworkTrafficData { PacketCount = 510, AveragePacketSize = 1900, Label = false },
      new NetworkTrafficData { PacketCount = 530, AveragePacketSize = 2000, Label = false },
      new NetworkTrafficData { PacketCount = 550, AveragePacketSize = 2100, Label = false },
      new NetworkTrafficData { PacketCount = 570, AveragePacketSize = 2200, Label = false },
      new NetworkTrafficData { PacketCount = 590, AveragePacketSize = 2300, Label = false },
      new NetworkTrafficData { PacketCount = 610, AveragePacketSize = 2400, Label = false }
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
