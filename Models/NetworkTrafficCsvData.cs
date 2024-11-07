using Microsoft.ML.Data;

public class NetworkTrafficCsvData
{
    [LoadColumn(0)] // Index of the Date column
    public string? Date { get; set; }

    [LoadColumn(1)] // Index of the LocalIp column
    public float LocalIp { get; set; }

    [LoadColumn(2)] // Index of the RemoteAsn column
    public float RemoteAsn { get; set; }

    [LoadColumn(3)] // Index of the Flows column
    public float Flows { get; set; } // Adjusted to float if needed

    [LoadColumn(4)] // Adjust the index to match your CSV file
    public bool Label { get; set; }
}
