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
    var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", 
    nameof(NetworkTrafficData.PacketCount), 
    nameof(NetworkTrafficData.AveragePacketSize),
    nameof(NetworkTrafficData.PacketDuration),
    nameof(NetworkTrafficData.IntervalBetweenPackets))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"));

    // Trainer
    var trainer = mlContext.BinaryClassification.Trainers.FastTree(
    labelColumnName: "Label",
    featureColumnName: "Features",
    numberOfLeaves: 15,       
    learningRate: 0.01,       
    numberOfTrees: 300        
    );

    // Train and evaluate model
    var trainingPipeline = dataProcessPipeline.Append(trainer);

    
    // Cross validation 
    var cvResults = mlContext.BinaryClassification.CrossValidate(data, trainingPipeline, numberOfFolds: 3);
    var avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
    var avgAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);
    var avgF1Score = cvResults.Average(r => r.Metrics.F1Score);

    Console.WriteLine($"Cross-validated Model accuracy: {avgAccuracy:P2}");
    Console.WriteLine($"AUC: {avgAuc:P2}");
    Console.WriteLine($"F1 Score: {avgF1Score:P2}");


    var model = trainingPipeline.Fit(data);

    // Save the Model
    mlContext.Model.Save(model, data.Schema, "NetworkAnomalyDetectionModel.zip");

    EvaluateModel(mlContext, model, data);

    // custom threshold
    float anomalyThreshold = 0.6f; 
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

  //Fake data for training
  static IDataView LoadData(MLContext mlContext)
{
  var data = new[]
  {
    // Normal traffic samples
    new NetworkTrafficData { PacketCount = 100, AveragePacketSize = 500, PacketDuration = 100, IntervalBetweenPackets = 50, Label = true },
    new NetworkTrafficData { PacketCount = 120, AveragePacketSize = 600, PacketDuration = 110, IntervalBetweenPackets = 52, Label = true },
    new NetworkTrafficData { PacketCount = 130, AveragePacketSize = 750, PacketDuration = 115, IntervalBetweenPackets = 60, Label = true },
    new NetworkTrafficData { PacketCount = 110, AveragePacketSize = 680, PacketDuration = 105, IntervalBetweenPackets = 55, Label = true },
    new NetworkTrafficData { PacketCount = 140, AveragePacketSize = 700, PacketDuration = 120, IntervalBetweenPackets = 65, Label = true },
    new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 520, PacketDuration = 130, IntervalBetweenPackets = 57, Label = true },
    new NetworkTrafficData { PacketCount = 160, AveragePacketSize = 710, PacketDuration = 125, IntervalBetweenPackets = 58, Label = true },
    new NetworkTrafficData { PacketCount = 170, AveragePacketSize = 730, PacketDuration = 140, IntervalBetweenPackets = 70, Label = true },
    new NetworkTrafficData { PacketCount = 180, AveragePacketSize = 820, PacketDuration = 135, IntervalBetweenPackets = 60, Label = true },
    new NetworkTrafficData { PacketCount = 200, AveragePacketSize = 900, PacketDuration = 150, IntervalBetweenPackets = 75, Label = true },
    new NetworkTrafficData { PacketCount = 210, AveragePacketSize = 950, PacketDuration = 145, IntervalBetweenPackets = 78, Label = true },
    new NetworkTrafficData { PacketCount = 220, AveragePacketSize = 980, PacketDuration = 155, IntervalBetweenPackets = 80, Label = true },
    new NetworkTrafficData { PacketCount = 230, AveragePacketSize = 1000, PacketDuration = 160, IntervalBetweenPackets = 82, Label = true },
    new NetworkTrafficData { PacketCount = 240, AveragePacketSize = 1050, PacketDuration = 170, IntervalBetweenPackets = 85, Label = true },
    new NetworkTrafficData { PacketCount = 250, AveragePacketSize = 1100, PacketDuration = 175, IntervalBetweenPackets = 88, Label = true },
    new NetworkTrafficData { PacketCount = 260, AveragePacketSize = 1150, PacketDuration = 180, IntervalBetweenPackets = 90, Label = true },
    new NetworkTrafficData { PacketCount = 270, AveragePacketSize = 1200, PacketDuration = 185, IntervalBetweenPackets = 92, Label = true },
    new NetworkTrafficData { PacketCount = 280, AveragePacketSize = 1250, PacketDuration = 190, IntervalBetweenPackets = 95, Label = true },
    new NetworkTrafficData { PacketCount = 290, AveragePacketSize = 1300, PacketDuration = 195, IntervalBetweenPackets = 98, Label = true },
    new NetworkTrafficData { PacketCount = 300, AveragePacketSize = 1350, PacketDuration = 200, IntervalBetweenPackets = 100, Label = true },
    new NetworkTrafficData { PacketCount = 310, AveragePacketSize = 1400, PacketDuration = 205, IntervalBetweenPackets = 102, Label = true },
    new NetworkTrafficData { PacketCount = 320, AveragePacketSize = 1450, PacketDuration = 210, IntervalBetweenPackets = 105, Label = true },
    new NetworkTrafficData { PacketCount = 330, AveragePacketSize = 1500, PacketDuration = 215, IntervalBetweenPackets = 108, Label = true },
    new NetworkTrafficData { PacketCount = 340, AveragePacketSize = 1550, PacketDuration = 220, IntervalBetweenPackets = 110, Label = true },
    new NetworkTrafficData { PacketCount = 350, AveragePacketSize = 1600, PacketDuration = 225, IntervalBetweenPackets = 112, Label = true },

    // Anomalous traffic samples
    new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500, PacketDuration = 300, IntervalBetweenPackets = 200, Label = false },
    new NetworkTrafficData { PacketCount = 420, AveragePacketSize = 1600, PacketDuration = 320, IntervalBetweenPackets = 210, Label = false },
    new NetworkTrafficData { PacketCount = 430, AveragePacketSize = 1700, PacketDuration = 330, IntervalBetweenPackets = 220, Label = false },
    new NetworkTrafficData { PacketCount = 440, AveragePacketSize = 1800, PacketDuration = 340, IntervalBetweenPackets = 230, Label = false },
    new NetworkTrafficData { PacketCount = 450, AveragePacketSize = 1900, PacketDuration = 350, IntervalBetweenPackets = 240, Label = false },
    new NetworkTrafficData { PacketCount = 460, AveragePacketSize = 2000, PacketDuration = 360, IntervalBetweenPackets = 250, Label = false },
    new NetworkTrafficData { PacketCount = 470, AveragePacketSize = 2100, PacketDuration = 370, IntervalBetweenPackets = 260, Label = false },
    new NetworkTrafficData { PacketCount = 480, AveragePacketSize = 2200, PacketDuration = 380, IntervalBetweenPackets = 270, Label = false },
    new NetworkTrafficData { PacketCount = 490, AveragePacketSize = 2300, PacketDuration = 390, IntervalBetweenPackets = 280, Label = false },
    new NetworkTrafficData { PacketCount = 500, AveragePacketSize = 2400, PacketDuration = 400, IntervalBetweenPackets = 290, Label = false },
    new NetworkTrafficData { PacketCount = 510, AveragePacketSize = 2500, PacketDuration = 410, IntervalBetweenPackets = 300, Label = false },
    new NetworkTrafficData { PacketCount = 520, AveragePacketSize = 2600, PacketDuration = 420, IntervalBetweenPackets = 310, Label = false },
    new NetworkTrafficData { PacketCount = 530, AveragePacketSize = 2700, PacketDuration = 430, IntervalBetweenPackets = 320, Label = false },
    new NetworkTrafficData { PacketCount = 540, AveragePacketSize = 2800, PacketDuration = 440, IntervalBetweenPackets = 330, Label = false },
    new NetworkTrafficData { PacketCount = 550, AveragePacketSize = 2900, PacketDuration = 450, IntervalBetweenPackets = 340, Label = false },
    new NetworkTrafficData { PacketCount = 560, AveragePacketSize = 3000, PacketDuration = 460, IntervalBetweenPackets = 350, Label = false },
    new NetworkTrafficData { PacketCount = 570, AveragePacketSize = 3100, PacketDuration = 470, IntervalBetweenPackets = 360, Label = false },
    new NetworkTrafficData { PacketCount = 580, AveragePacketSize = 3200, PacketDuration = 480, IntervalBetweenPackets = 370, Label = false },
    new NetworkTrafficData { PacketCount = 590, AveragePacketSize = 3300, PacketDuration = 490, IntervalBetweenPackets = 380, Label = false },
    new NetworkTrafficData { PacketCount = 600, AveragePacketSize = 3400, PacketDuration = 500, IntervalBetweenPackets = 390, Label = false }
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
