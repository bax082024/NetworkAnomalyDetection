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
  }


}