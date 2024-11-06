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
    nameof(NetworkTrafficData.IntervalBetweenPackets),
    nameof(NetworkTrafficData.PacketFrequency),
    nameof(NetworkTrafficData.TotalDataSent),
    nameof(NetworkTrafficData.SourceDestinationRatio))
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
      // Normal traffic samples
      new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 500, PacketDuration = 100, IntervalBetweenPackets = 50, PacketFrequency = 5, TotalDataSent = 2500, SourceDestinationRatio = 1.0f },
      new NetworkTrafficData { PacketCount = 180, AveragePacketSize = 700, PacketDuration = 120, IntervalBetweenPackets = 55, PacketFrequency = 6, TotalDataSent = 3000, SourceDestinationRatio = 1.2f },
      new NetworkTrafficData { PacketCount = 220, AveragePacketSize = 900, PacketDuration = 130, IntervalBetweenPackets = 60, PacketFrequency = 7, TotalDataSent = 4000, SourceDestinationRatio = 1.5f },
      new NetworkTrafficData { PacketCount = 250, AveragePacketSize = 1000, PacketDuration = 140, IntervalBetweenPackets = 65, PacketFrequency = 8, TotalDataSent = 5000, SourceDestinationRatio = 1.8f },
      
      // Anomalous traffic samples
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500, PacketDuration = 300, IntervalBetweenPackets = 200, PacketFrequency = 15, TotalDataSent = 7500, SourceDestinationRatio = 3.5f },
      new NetworkTrafficData { PacketCount = 500, AveragePacketSize = 2000, PacketDuration = 400, IntervalBetweenPackets = 250, PacketFrequency = 17, TotalDataSent = 10000, SourceDestinationRatio = 4.0f },
      new NetworkTrafficData { PacketCount = 600, AveragePacketSize = 2500, PacketDuration = 450, IntervalBetweenPackets = 300, PacketFrequency = 19, TotalDataSent = 12500, SourceDestinationRatio = 4.5f },
      new NetworkTrafficData { PacketCount = 700, AveragePacketSize = 2900, PacketDuration = 500, IntervalBetweenPackets = 350, PacketFrequency = 21, TotalDataSent = 15000, SourceDestinationRatio = 5.0f }
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
      // Normal samples
      new NetworkTrafficData { PacketCount = 100, AveragePacketSize = 500, PacketDuration = 100, IntervalBetweenPackets = 50, PacketFrequency = 5, TotalDataSent = 2500, SourceDestinationRatio = 1, Label = true },
      new NetworkTrafficData { PacketCount = 120, AveragePacketSize = 600, PacketDuration = 120, IntervalBetweenPackets = 55, PacketFrequency = 4, TotalDataSent = 2880, SourceDestinationRatio = 1.2f, Label = true },
      new NetworkTrafficData { PacketCount = 130, AveragePacketSize = 750, PacketDuration = 150, IntervalBetweenPackets = 60, PacketFrequency = 4.5f, TotalDataSent = 3250, SourceDestinationRatio = 1.1f, Label = true },
      new NetworkTrafficData { PacketCount = 110, AveragePacketSize = 680, PacketDuration = 115, IntervalBetweenPackets = 52, PacketFrequency = 5.2f, TotalDataSent = 2860, SourceDestinationRatio = 1, Label = true },
      new NetworkTrafficData { PacketCount = 150, AveragePacketSize = 520, PacketDuration = 125, IntervalBetweenPackets = 57, PacketFrequency = 3.8f, TotalDataSent = 3120, SourceDestinationRatio = 1.3f, Label = true },
      new NetworkTrafficData { PacketCount = 160, AveragePacketSize = 700, PacketDuration = 130, IntervalBetweenPackets = 65, PacketFrequency = 5.5f, TotalDataSent = 3600, SourceDestinationRatio = 1.4f, Label = true },
      new NetworkTrafficData { PacketCount = 180, AveragePacketSize = 800, PacketDuration = 140, IntervalBetweenPackets = 58, PacketFrequency = 5.8f, TotalDataSent = 4000, SourceDestinationRatio = 1.1f, Label = true },
      new NetworkTrafficData { PacketCount = 140, AveragePacketSize = 750, PacketDuration = 110, IntervalBetweenPackets = 56, PacketFrequency = 4.9f, TotalDataSent = 3250, SourceDestinationRatio = 1.3f, Label = true },
      new NetworkTrafficData { PacketCount = 200, AveragePacketSize = 1000, PacketDuration = 180, IntervalBetweenPackets = 60, PacketFrequency = 3.2f, TotalDataSent = 5000, SourceDestinationRatio = 1.0f, Label = true },
      new NetworkTrafficData { PacketCount = 220, AveragePacketSize = 1050, PacketDuration = 195, IntervalBetweenPackets = 65, PacketFrequency = 2.9f, TotalDataSent = 5250, SourceDestinationRatio = 1.2f, Label = true },

      // normal samples
      new NetworkTrafficData { PacketCount = 135, AveragePacketSize = 700, PacketDuration = 140, IntervalBetweenPackets = 57, PacketFrequency = 5.6f, TotalDataSent = 3100, SourceDestinationRatio = 1.15f, Label = true },
      new NetworkTrafficData { PacketCount = 145, AveragePacketSize = 720, PacketDuration = 150, IntervalBetweenPackets = 59, PacketFrequency = 5.7f, TotalDataSent = 3240, SourceDestinationRatio = 1.17f, Label = true },
      new NetworkTrafficData { PacketCount = 170, AveragePacketSize = 760, PacketDuration = 160, IntervalBetweenPackets = 55, PacketFrequency = 5.2f, TotalDataSent = 3600, SourceDestinationRatio = 1.1f, Label = true },
      new NetworkTrafficData { PacketCount = 155, AveragePacketSize = 680, PacketDuration = 135, IntervalBetweenPackets = 53, PacketFrequency = 5.4f, TotalDataSent = 3300, SourceDestinationRatio = 1.05f, Label = true },
      new NetworkTrafficData { PacketCount = 195, AveragePacketSize = 800, PacketDuration = 175, IntervalBetweenPackets = 62, PacketFrequency = 4.8f, TotalDataSent = 3900, SourceDestinationRatio = 1.0f, Label = true },
      new NetworkTrafficData { PacketCount = 205, AveragePacketSize = 900, PacketDuration = 185, IntervalBetweenPackets = 60, PacketFrequency = 3.9f, TotalDataSent = 5000, SourceDestinationRatio = 1.3f, Label = true },
      new NetworkTrafficData { PacketCount = 210, AveragePacketSize = 850, PacketDuration = 160, IntervalBetweenPackets = 58, PacketFrequency = 4.1f, TotalDataSent = 5150, SourceDestinationRatio = 1.4f, Label = true },
      new NetworkTrafficData { PacketCount = 175, AveragePacketSize = 670, PacketDuration = 145, IntervalBetweenPackets = 54, PacketFrequency = 4.9f, TotalDataSent = 3360, SourceDestinationRatio = 1.2f, Label = true },
      new NetworkTrafficData { PacketCount = 225, AveragePacketSize = 880, PacketDuration = 170, IntervalBetweenPackets = 61, PacketFrequency = 3.5f, TotalDataSent = 5600, SourceDestinationRatio = 1.1f, Label = true },
      new NetworkTrafficData { PacketCount = 190, AveragePacketSize = 780, PacketDuration = 165, IntervalBetweenPackets = 63, PacketFrequency = 4.0f, TotalDataSent = 4300, SourceDestinationRatio = 1.15f, Label = true },

      // Anomaly samples
      new NetworkTrafficData { PacketCount = 400, AveragePacketSize = 1500, PacketDuration = 300, IntervalBetweenPackets = 200, PacketFrequency = 10, TotalDataSent = 6000, SourceDestinationRatio = 3, Label = false },
      new NetworkTrafficData { PacketCount = 450, AveragePacketSize = 1600, PacketDuration = 350, IntervalBetweenPackets = 220, PacketFrequency = 11, TotalDataSent = 7200, SourceDestinationRatio = 2.8f, Label = false },
      new NetworkTrafficData { PacketCount = 470, AveragePacketSize = 1700, PacketDuration = 400, IntervalBetweenPackets = 250, PacketFrequency = 12, TotalDataSent = 7650, SourceDestinationRatio = 3.2f, Label = false },
      new NetworkTrafficData { PacketCount = 490, AveragePacketSize = 1800, PacketDuration = 420, IntervalBetweenPackets = 260, PacketFrequency = 13, TotalDataSent = 8000, SourceDestinationRatio = 3.5f, Label = false },
      new NetworkTrafficData { PacketCount = 510, AveragePacketSize = 1900, PacketDuration = 430, IntervalBetweenPackets = 270, PacketFrequency = 14, TotalDataSent = 8500, SourceDestinationRatio = 3.3f, Label = false },
      
      // Anomaly samples
      new NetworkTrafficData { PacketCount = 530, AveragePacketSize = 2000, PacketDuration = 440, IntervalBetweenPackets = 280, PacketFrequency = 14.5f, TotalDataSent = 9000, SourceDestinationRatio = 3.4f, Label = false },
      new NetworkTrafficData { PacketCount = 550, AveragePacketSize = 2100, PacketDuration = 450, IntervalBetweenPackets = 290, PacketFrequency = 15, TotalDataSent = 10000, SourceDestinationRatio = 3.8f, Label = false },
      new NetworkTrafficData { PacketCount = 570, AveragePacketSize = 2200, PacketDuration = 460, IntervalBetweenPackets = 300, PacketFrequency = 16, TotalDataSent = 10500, SourceDestinationRatio = 3.9f, Label = false },
      new NetworkTrafficData { PacketCount = 590, AveragePacketSize = 2300, PacketDuration = 470, IntervalBetweenPackets = 310, PacketFrequency = 16.5f, TotalDataSent = 11000, SourceDestinationRatio = 4.0f, Label = false },
      new NetworkTrafficData { PacketCount = 610, AveragePacketSize = 2400, PacketDuration = 480, IntervalBetweenPackets = 320, PacketFrequency = 17, TotalDataSent = 11500, SourceDestinationRatio = 4.2f, Label = false },

      new NetworkTrafficData { PacketCount = 620, AveragePacketSize = 2500, PacketDuration = 490, IntervalBetweenPackets = 330, PacketFrequency = 17.5f, TotalDataSent = 12000, SourceDestinationRatio = 4.5f, Label = false },
      new NetworkTrafficData { PacketCount = 640, AveragePacketSize = 2600, PacketDuration = 500, IntervalBetweenPackets = 340, PacketFrequency = 18, TotalDataSent = 12500, SourceDestinationRatio = 4.6f, Label = false },
      new NetworkTrafficData { PacketCount = 660, AveragePacketSize = 2700, PacketDuration = 510, IntervalBetweenPackets = 350, PacketFrequency = 19, TotalDataSent = 13000, SourceDestinationRatio = 4.7f, Label = false },
      new NetworkTrafficData { PacketCount = 680, AveragePacketSize = 2800, PacketDuration = 520, IntervalBetweenPackets = 360, PacketFrequency = 20, TotalDataSent = 13500, SourceDestinationRatio = 4.8f, Label = false },
      new NetworkTrafficData { PacketCount = 700, AveragePacketSize = 2900, PacketDuration = 530, IntervalBetweenPackets = 370, PacketFrequency = 20.5f, TotalDataSent = 14000, SourceDestinationRatio = 5.0f, Label = false }
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
