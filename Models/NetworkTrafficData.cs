using Microsoft.ML.Data;

public class NetworkTrafficData
{
  [LoadColumn(0)]
  public float PacketCount { get; set; }

  [LoadColumn(1)]
  public float AveragePacketSize { get; set; }

  [LoadColumn(2)]
  public float PacketDuration { get; set; } // true for normal traffic and false for anomaly

  [LoadColumn(3)]
  public float IntervalBetweenPackets { get; set; }

  [LoadColumn(4)]
  public bool Label { get; set; }
}