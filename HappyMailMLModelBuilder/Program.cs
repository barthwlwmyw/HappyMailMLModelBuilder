using Microsoft.ML;
using Microsoft.ML.Data;

namespace HappyMailMLModelBuilder
{
    class Program
    {
        private static string TRAIN_DATA_FILEPATH = @".\wikipedia-detox-250-line-data.tsv";
        private static string MODEL_FILEPATH = @".\MLModel.zip";

        private static MLContext mlContext = new MLContext(seed: 1);

        static void Main(string[] args)
        {
            // Load data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: '\t');

            // Transform data and set learning algorithm
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            // Train model
            ITransformer mlModel = trainingPipeline.Fit(trainingDataView);

            // Save model
            mlContext.Model.Save(mlModel, trainingDataView.Schema, MODEL_FILEPATH);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Sentiment", "Sentiment")
                                      .Append(mlContext.Transforms.Text.FeaturizeText("SentimentText_tf", "SentimentText"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "SentimentText_tf"))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Sentiment", featureColumnName: "Features")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }
    }

    public class ModelInput
    {
        [ColumnName("Sentiment"), LoadColumn(0)]
        public string Sentiment { get; set; }

        [ColumnName("SentimentText"), LoadColumn(1)]
        public string SentimentText { get; set; }

        [ColumnName("LoggedIn"), LoadColumn(2)]
        public string LoggedIn { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }

        public float[] Score { get; set; }
    }
}
