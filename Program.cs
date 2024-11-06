using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

class Program
{
  static void Main(string[] args)
  {
    // initialize MLContext
    MLContext mlContext = new MLContext();

    // Load data
    var data = LoadData(mlContext);

    // Define processing pipeline
    var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", nameof(NetworkTrafficData.PacketCount), )

    // Define trainer
    var trainer = mlContext.BinaryClassification.Trainers.FastTree(
      labelColumnName: "Label",
      featureColumnName: "Features",
      numberOfLeaves: 50,
      learningRate: 0.05,
      numberOfTrees: 200
    );

    var trainingPipeline = dataProcessPipeline.Append(trainer);

    // Train Model
    var model = trainingPipeline.Fit(data);
  }


}