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

        

        // Load the CSV data
        string dataPath = "Data/cs448b_ipasn.csv";
        IDataView dataView = mlContext.Data.LoadFromTextFile<NetworkTrafficCsvData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ','
        );

        

        List<NetworkTrafficCsvData> dataList = mlContext.Data.CreateEnumerable<NetworkTrafficCsvData>(dataView, reuseRowObject: false).ToList();

        // Adding several positive samples
        dataList.Add(new NetworkTrafficCsvData { Date = "2006-07-01", LocalIp = 2, RemoteAsn = 701, Flows = 50, Label = true });
        dataList.Add(new NetworkTrafficCsvData { Date = "2006-07-02", LocalIp = 3, RemoteAsn = 1239, Flows = 60, Label = true });
        dataList.Add(new NetworkTrafficCsvData { Date = "2006-07-03", LocalIp = 4, RemoteAsn = 16755, Flows = 80, Label = true });
        dataList.Add(new NetworkTrafficCsvData { Date = "2006-07-04", LocalIp = 5, RemoteAsn = 3561, Flows = 100, Label = true });
        dataList.Add(new NetworkTrafficCsvData { Date = "2006-07-05", LocalIp = 6, RemoteAsn = 7132, Flows = 120, Label = true });


        // Duplicate positive samples to balance the dataset
        var positiveSamples = dataList.Where(d => d.Label == true).ToList();
        for (int i = 0; i < 4000; i++) // Duplicate enough times to get around 5,000 positive samples
        {
            dataList.AddRange(positiveSamples);
        }

        // Reload the data view with the modified list
        dataView = mlContext.Data.LoadFromEnumerable(dataList);

        // Count positive and negative samples
        int positiveCount = dataList.Count(d => d.Label == true);
        int negativeCount = dataList.Count(d => d.Label == false);

        Console.WriteLine($"Positive samples: {positiveCount}");
        Console.WriteLine($"Negative samples: {negativeCount}");


        // Display a few rows to verify it loaded correctly
        var preview = dataView.Preview();
        foreach (var row in preview.RowView)
        {
            foreach (var kv in row.Values)
            {
                Console.Write($"{kv.Key}: {kv.Value} | ");
            }
            Console.WriteLine();
        }

        // Processing Pipeline
        var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", 
            nameof(NetworkTrafficCsvData.LocalIp), 
            nameof(NetworkTrafficCsvData.RemoteAsn),
            nameof(NetworkTrafficCsvData.Flows))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));



        // var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        // labelColumnName: "Label",
        // featureColumnName: "Features");


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

        // Cross-validation
        var cvResults = mlContext.BinaryClassification.CrossValidate(dataView, trainingPipeline, numberOfFolds: 3);
        var avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
        var avgAuc = cvResults.Average(r => r.Metrics.AreaUnderRocCurve);
        var avgF1Score = cvResults.Average(r => r.Metrics.F1Score);

        Console.WriteLine($"Cross-validated Model accuracy: {avgAccuracy:P2}");
        Console.WriteLine($"AUC: {avgAuc:P2}");
        Console.WriteLine($"F1 Score: {avgF1Score:P2}");

        var model = trainingPipeline.Fit(dataView);

        // Save the Model
        mlContext.Model.Save(model, dataView.Schema, "NetworkAnomalyDetectionModel.zip");

        // Comment out or add EvaluateModel function
        EvaluateModel(mlContext, model, dataView);
    }

    // Define EvaluateModel if you want to use it
    static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data)
    {
        var predictions = model.Transform(data);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

        Console.WriteLine($"Model accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
    }
}
