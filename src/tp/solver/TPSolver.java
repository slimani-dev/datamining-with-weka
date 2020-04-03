package tp.solver;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class TPSolver {
    public static void doTheTP(FileWriter fileWriter, ArrayList<String> datasetPaths, ArrayList<Classifier> classifiers) throws Exception {
        ConverterUtils.DataSource source;

        for (String datasetPath : datasetPaths) {

            source = new ConverterUtils.DataSource(datasetPath);
            Instances dataset = source.getDataSet();

            dataSetInformation(fileWriter, dataset);

            doTheTP(fileWriter, classifiers, dataset);

        }
    }

    public static void doTheTP(FileWriter fileWriter, ArrayList<Classifier> classifiers, Instances dataset) throws Exception {

        double bestPrecisionForAll = 0;
        double bestRecallForAll = 0;
        double bestFMeasureForAll = 0;
        double bestKappaForAll = 0;
        double bestPctCorrectForAll = 0;


        String bestPrecisionMethodForAll = "";
        String bestRecallMethodForAll = "";
        String bestFMeasureMethodForAll = "";
        String bestKappaMethodForAll = "";
        String bestPctCorrectMethodForAll = "";


        String bestPrecisionAlgorithmForAll = "";
        String bestRecallAlgorithmForAll = "";
        String bestFMeasureAlgorithmForAll = "";
        String bestKappaAlgorithmForAll = "";
        String bestPctCorrectAlgorithmForAll = "";


        for (Classifier classifier : classifiers) {
            try {
                double bestPrecision = 0;
                double bestRecall = 0;
                double bestFMeasure = 0;
                double bestKappa = 0;
                double bestPctCorrect = 0;

                String bestPrecisionMethod = "";
                String bestRecallMethod = "";
                String bestFMeasureMethod = "";
                String bestKappaMethod = "";
                String bestPctCorrectMethod = "";

                Evaluation eval;

                System.out.println("\t" + classifier.getClass().getSimpleName());
                fileWriter.write("\n\n");
                fileWriter.write("#### Algo: " + classifier.getClass().getSimpleName());
                fileWriter.write("\n\n");


                // cross validation
                eval = new Evaluation(dataset);
                eval.crossValidateModel(classifier, dataset, 10, new Random(1));

                double CrossValidationWeightedPrecision = eval.weightedPrecision();
                double CrossValidationWeightedRecall = eval.weightedRecall();
                double CrossValidationWeightedFMeasure = eval.weightedFMeasure();
                double CrossValidationWeightedKappa = eval.kappa();
                double CrossValidationWeightedPctCorrect = eval.pctCorrect();


                // leave one out
                eval = new Evaluation(dataset);
                eval.crossValidateModel(classifier, dataset, dataset.numInstances(), new Random(1));

                double CrossValidationAllWeightedPrecision = eval.weightedPrecision();
                double CrossValidationAllWeightedRecall = eval.weightedRecall();
                double CrossValidationAllWeightedFMeasure = eval.weightedFMeasure();
                double CrossValidationAllWeightedKappa = eval.kappa();
                double CrossValidationAllWeightedPctCorrect = eval.pctCorrect();

                // hold out
                int percent = 66;

                dataset.randomize(new Random(1));

                RemovePercentage rp = new RemovePercentage();
                rp.setPercentage(percent);
                rp.setInvertSelection(true);
                rp.setInputFormat(dataset);
                Instances train = Filter.useFilter(dataset, rp);

                rp.setPercentage(percent);
                rp.setInvertSelection(false);
                rp.setInputFormat(dataset);
                Instances test = Filter.useFilter(dataset, rp);

                classifier.buildClassifier(train);
                eval = new Evaluation(train);
                eval.evaluateModel(classifier, test);

                double PercentageSplit66WeightedPrecision = eval.weightedPrecision();
                double PercentageSplit66WeightedRecall = eval.weightedRecall();
                double PercentageSplit66WeightedFMeasure = eval.weightedFMeasure();
                double PercentageSplit66WeightedKappa = eval.kappa();
                double PercentageSplit66WeightedPctCorrect = eval.pctCorrect();

                // base d'apprentissage

                classifier.buildClassifier(dataset);
                eval = new Evaluation(dataset);
                eval.evaluateModel(classifier, dataset);

                double TrainingSetWeightedPrecision = eval.weightedPrecision();
                double TrainingSetWeightedRecall = eval.weightedRecall();
                double TrainingSetWeightedFMeasure = eval.weightedFMeasure();
                double TrainingSetWeightedKappa = eval.kappa();
                double TrainingSetWeightedPctCorrect = eval.pctCorrect();


                if (bestPrecision < CrossValidationWeightedPrecision) {
                    bestPrecision = CrossValidationWeightedPrecision;
                    bestPrecisionMethod = "Cross validation (10)";
                }
                if (bestRecall < CrossValidationWeightedRecall) {
                    bestRecall = CrossValidationWeightedRecall;
                    bestRecallMethod = "Cross validation (10)";
                }
                if (bestFMeasure < CrossValidationWeightedFMeasure) {
                    bestFMeasure = CrossValidationWeightedFMeasure;
                    bestFMeasureMethod = "Cross validation (10)";
                }
                if (bestKappa < CrossValidationWeightedKappa) {
                    bestKappa = CrossValidationWeightedKappa;
                    bestKappaMethod = "Cross validation (10)";
                }
                if (bestPctCorrect < CrossValidationWeightedPctCorrect) {
                    bestPctCorrect = CrossValidationWeightedPctCorrect;
                    bestPctCorrectMethod = "Cross validation (10)";
                }


                if (bestPrecision < CrossValidationAllWeightedPrecision) {
                    bestPrecision = CrossValidationAllWeightedPrecision;
                    bestPrecisionMethod = "Leave one out (" + dataset.numInstances() + ")";
                }
                if (bestRecall < CrossValidationAllWeightedRecall) {
                    bestRecall = CrossValidationAllWeightedRecall;
                    bestRecallMethod = "Leave one out (" + dataset.numInstances() + ")";
                }
                if (bestFMeasure < CrossValidationAllWeightedFMeasure) {
                    bestFMeasure = CrossValidationAllWeightedFMeasure;
                    bestFMeasureMethod = "Leave one out (" + dataset.numInstances() + ")";
                }
                if (bestKappa < CrossValidationAllWeightedKappa) {
                    bestKappa = CrossValidationAllWeightedKappa;
                    bestKappaMethod = "Leave one out (" + dataset.numInstances() + ")";
                }
                if (bestPctCorrect < CrossValidationAllWeightedPctCorrect) {
                    bestPctCorrect = CrossValidationAllWeightedPctCorrect;
                    bestPctCorrectMethod = "Leave one out (" + dataset.numInstances() + ")";
                }

                if (bestPrecision < PercentageSplit66WeightedPrecision) {
                    bestPrecision = PercentageSplit66WeightedPrecision;
                    bestPrecisionMethod = "Hold out (" + percent + "%)";
                }
                if (bestRecall < PercentageSplit66WeightedRecall) {
                    bestRecall = PercentageSplit66WeightedRecall;
                    bestRecallMethod = "Hold out (" + percent + "%)";
                }
                if (bestFMeasure < PercentageSplit66WeightedFMeasure) {
                    bestFMeasure = PercentageSplit66WeightedFMeasure;
                    bestFMeasureMethod = "Hold out (" + percent + "%)";
                }
                if (bestKappa < PercentageSplit66WeightedKappa) {
                    bestKappa = PercentageSplit66WeightedKappa;
                    bestKappaMethod = "Hold out (" + percent + "%)";
                }
                if (bestPctCorrect < PercentageSplit66WeightedPctCorrect) {
                    bestPctCorrect = PercentageSplit66WeightedPctCorrect;
                    bestPctCorrectMethod = "Hold out (" + percent + "%)";
                }



                fileWriter.write("**Results** :");

                fileWriter.write("\n\n");

                fileWriter.write("| | Kapp | PctCorrect | Precision | Recall | F-Measure |\n" +
                        "| --- | --- | --- | --- | --- | --- |\n");

                fileWriter.write("| Training set : | " +
                        String.format("%.3f", TrainingSetWeightedKappa) + " | " +
                        String.format("%.3f", TrainingSetWeightedPctCorrect) + " | " +
                        String.format("%.3f", TrainingSetWeightedPrecision) + " | " +
                        String.format("%.3f", TrainingSetWeightedRecall) + " | " +
                        String.format("%.3f", TrainingSetWeightedFMeasure) + " | \n");

                fileWriter.write("| Cross-validation (10) : | " +
                        String.format("%.3f", CrossValidationWeightedKappa) + " | " +
                        String.format("%.3f", CrossValidationWeightedPctCorrect) + " | " +
                        String.format("%.3f", CrossValidationWeightedPrecision) + " | " +
                        String.format("%.3f", CrossValidationWeightedRecall) + " | " +
                        String.format("%.3f", CrossValidationWeightedFMeasure) + " | \n");

                fileWriter.write("| Leave one out (" + dataset.numInstances() + ") : | " +
                        String.format("%.3f", CrossValidationAllWeightedKappa) + " | " +
                        String.format("%.3f", CrossValidationAllWeightedPctCorrect) + " | " +
                        String.format("%.3f", CrossValidationAllWeightedPrecision) + " | " +
                        String.format("%.3f", CrossValidationAllWeightedRecall) + " | " +
                        String.format("%.3f", CrossValidationAllWeightedFMeasure) + " | \n");

                fileWriter.write("| Hold out (" + percent + "%) : | " +
                        String.format("%.3f", PercentageSplit66WeightedKappa) + " | " +
                        String.format("%.3f", PercentageSplit66WeightedPctCorrect) + " | " +
                        String.format("%.3f", PercentageSplit66WeightedPrecision) + " | " +
                        String.format("%.3f", PercentageSplit66WeightedRecall) + " | " +
                        String.format("%.3f", PercentageSplit66WeightedFMeasure) + " | \n");




                fileWriter.write("**Comparaison** :\n\n" +
                        "comparaison à l'aide du **Kappa** :\n\n" +
                        "le meilleur résultat est : " + bestKappaMethod + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestKappa) + "\n\n" +
                        "comparaison à l'aide du **pct Correct** :\n\n" +
                        "le meilleur résultat est : " + bestPctCorrectMethod + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestPctCorrect) + "\n\n" +
                        "comparaison à l'aide du **F-Measure** :\n\n" +
                        "le meilleur résultat est : " + bestFMeasureMethod + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestPrecision) + "\n\n" +
                        "comparaison à l'aide du **Recall** :\n\n" +
                        "le meilleur résultat est : " + bestRecallMethod + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestRecall) + "\n\n" +
                        "comparaison à l'aide du **Precision** :\n\n" +
                        "le meilleur résultat est " + bestPrecisionMethod + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestFMeasure) + "\n\n");

                if (bestPrecisionForAll < bestPrecision) {
                    bestPrecisionForAll = bestPrecision;
                    bestPrecisionMethodForAll = bestPrecisionMethod;
                    bestPrecisionAlgorithmForAll = classifier.getClass().getSimpleName();
                }
                if (bestRecallForAll < bestRecall) {
                    bestRecallForAll = bestRecall;
                    bestRecallMethodForAll = bestRecallMethod;
                    bestRecallAlgorithmForAll = classifier.getClass().getSimpleName();
                }
                if (bestFMeasureForAll < bestFMeasure) {
                    bestFMeasureForAll = bestFMeasure;
                    bestFMeasureMethodForAll = bestFMeasureMethod;
                    bestFMeasureAlgorithmForAll = classifier.getClass().getSimpleName();
                }
                if (bestKappaForAll < bestKappa) {
                    bestKappaForAll = bestKappa;
                    bestKappaMethodForAll = bestKappaMethod;
                    bestKappaAlgorithmForAll = classifier.getClass().getSimpleName();
                }
                if (bestPctCorrectForAll < bestPctCorrect) {
                    bestPctCorrectForAll = bestPctCorrect;
                    bestPctCorrectMethodForAll = bestPctCorrectMethod;
                    bestPctCorrectAlgorithmForAll = classifier.getClass().getSimpleName();
                }

            } catch (Exception e) {
                fileWriter.write("Une **erreur** s'est produite lors de l'évaluation\n\n");
                fileWriter.write("!>" + e.getMessage() + "\n\n");
            }

        }

        fileWriter.write("\n\n#### Comparaison entre algorithmes :\n\n" +
                "comparaison à l'aide du **Kappa** :\n\n" +
                "le meilleur algorithme est : **" + bestKappaAlgorithmForAll + "** \n\n" +
                "le meilleur résultat est : " + bestKappaMethodForAll + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestKappaForAll) + "\n\n\n" +
                "comparaison à l'aide du **F-Measure** :\n\n" +
                "le meilleur algorithme est : **" + bestPctCorrectAlgorithmForAll + "** \n\n" +
                "le meilleur résultat est : " + bestPctCorrectMethodForAll + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestPctCorrectForAll) + "\n\n\n" +
                "comparaison à l'aide du **F-Measure** :\n\n" +
                "le meilleur algorithme est : **" + bestFMeasureAlgorithmForAll + "** \n\n" +
                "le meilleur résultat est : " + bestFMeasureMethodForAll + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestFMeasureForAll) + "\n\n\n" +
                "comparaison à l'aide du **Recall** :\n\n" +
                "le meilleur algorithme est : **" + bestRecallAlgorithmForAll + "** \n\n" +
                "le meilleur résultat est : " + bestRecallMethodForAll + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestRecallForAll) + "\n\n\n" +
                "comparaison à l'aide du **Precision** :\n\n" +
                "le meilleur algorithme est : **" + bestPrecisionAlgorithmForAll + "** \n\n" +
                "le meilleur résultat est " + bestPrecisionMethodForAll + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestPrecisionForAll) + "\n\n\n");
    }

    public static void doTheTP(FileWriter fileWriter, Classifier classifier, Instances dataset) throws Exception {
        double bestPrecision = 0;
        double bestRecall = 0;
        double bestFMeasure = 0;
        double bestKappa = 0;
        double bestPctCorrect = 0;

        String bestPrecisionMethod = "";
        String bestRecallMethod = "";
        String bestFMeasureMethod = "";
        String bestKappaMethod = "";
        String bestPctCorrectMethod = "";

        Evaluation eval;

        System.out.println("\t" + classifier.getClass().getSimpleName());
        fileWriter.write("\n\n");
        fileWriter.write("#### Algo: " + classifier.getClass().getSimpleName());
        fileWriter.write("\n\n");


        // cross validation
        eval = new Evaluation(dataset);
        eval.crossValidateModel(classifier, dataset, 10, new Random(1));

        double CrossValidationWeightedPrecision = eval.weightedPrecision();
        double CrossValidationWeightedRecall = eval.weightedRecall();
        double CrossValidationWeightedFMeasure = eval.weightedFMeasure();
        double CrossValidationWeightedKappa = eval.kappa();
        double CrossValidationWeightedPctCorrect = eval.pctCorrect();


        // leave one out
        eval = new Evaluation(dataset);
        eval.crossValidateModel(classifier, dataset, dataset.numInstances(), new Random(1));

        double CrossValidationAllWeightedPrecision = eval.weightedPrecision();
        double CrossValidationAllWeightedRecall = eval.weightedRecall();
        double CrossValidationAllWeightedFMeasure = eval.weightedFMeasure();
        double CrossValidationAllWeightedKappa = eval.kappa();
        double CrossValidationAllWeightedPctCorrect = eval.pctCorrect();

        // hold out
        int percent = 66;

        dataset.randomize(new Random(1));

        RemovePercentage rp = new RemovePercentage();
        rp.setPercentage(percent);
        rp.setInvertSelection(true);
        rp.setInputFormat(dataset);
        Instances train = Filter.useFilter(dataset, rp);

        rp.setPercentage(percent);
        rp.setInvertSelection(false);
        rp.setInputFormat(dataset);
        Instances test = Filter.useFilter(dataset, rp);

        classifier.buildClassifier(train);
        eval = new Evaluation(train);
        eval.evaluateModel(classifier, test);

        double PercentageSplit66WeightedPrecision = eval.weightedPrecision();
        double PercentageSplit66WeightedRecall = eval.weightedRecall();
        double PercentageSplit66WeightedFMeasure = eval.weightedFMeasure();
        double PercentageSplit66WeightedKappa = eval.kappa();
        double PercentageSplit66WeightedPctCorrect = eval.pctCorrect();

        // base d'apprentissage

        classifier.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(classifier, dataset);

        double TrainingSetWeightedPrecision = eval.weightedPrecision();
        double TrainingSetWeightedRecall = eval.weightedRecall();
        double TrainingSetWeightedFMeasure = eval.weightedFMeasure();
        double TrainingSetWeightedKappa = eval.kappa();
        double TrainingSetWeightedPctCorrect = eval.pctCorrect();


        if (bestPrecision < CrossValidationWeightedPrecision) {
            bestPrecision = CrossValidationWeightedPrecision;
            bestPrecisionMethod = "Cross validation (10)";
        }
        if (bestRecall < CrossValidationWeightedRecall) {
            bestRecall = CrossValidationWeightedRecall;
            bestRecallMethod = "Cross validation (10)";
        }
        if (bestFMeasure < CrossValidationWeightedFMeasure) {
            bestFMeasure = CrossValidationWeightedFMeasure;
            bestFMeasureMethod = "Cross validation (10)";
        }
        if (bestKappa < CrossValidationWeightedKappa) {
            bestKappa = CrossValidationWeightedKappa;
            bestKappaMethod = "Cross validation (10)";
        }
        if (bestPctCorrect < CrossValidationWeightedPctCorrect) {
            bestPctCorrect = CrossValidationWeightedPctCorrect;
            bestPctCorrectMethod = "Cross validation (10)";
        }


        if (bestPrecision < CrossValidationAllWeightedPrecision) {
            bestPrecision = CrossValidationAllWeightedPrecision;
            bestPrecisionMethod = "Leave one out (" + dataset.numInstances() + ")";
        }
        if (bestRecall < CrossValidationAllWeightedRecall) {
            bestRecall = CrossValidationAllWeightedRecall;
            bestRecallMethod = "Leave one out (" + dataset.numInstances() + ")";
        }
        if (bestFMeasure < CrossValidationAllWeightedFMeasure) {
            bestFMeasure = CrossValidationAllWeightedFMeasure;
            bestFMeasureMethod = "Leave one out (" + dataset.numInstances() + ")";
        }
        if (bestKappa < CrossValidationAllWeightedKappa) {
            bestKappa = CrossValidationAllWeightedKappa;
            bestKappaMethod = "Leave one out (" + dataset.numInstances() + ")";
        }
        if (bestPctCorrect < CrossValidationAllWeightedPctCorrect) {
            bestPctCorrect = CrossValidationAllWeightedPctCorrect;
            bestPctCorrectMethod = "Leave one out (" + dataset.numInstances() + ")";
        }

        if (bestPrecision < PercentageSplit66WeightedPrecision) {
            bestPrecision = PercentageSplit66WeightedPrecision;
            bestPrecisionMethod = "Hold out (" + percent + "%)";
        }
        if (bestRecall < PercentageSplit66WeightedRecall) {
            bestRecall = PercentageSplit66WeightedRecall;
            bestRecallMethod = "Hold out (" + percent + "%)";
        }
        if (bestFMeasure < PercentageSplit66WeightedFMeasure) {
            bestFMeasure = PercentageSplit66WeightedFMeasure;
            bestFMeasureMethod = "Hold out (" + percent + "%)";
        }
        if (bestKappa < PercentageSplit66WeightedKappa) {
            bestKappa = PercentageSplit66WeightedKappa;
            bestKappaMethod = "Hold out (" + percent + "%)";
        }
        if (bestPctCorrect < PercentageSplit66WeightedPctCorrect) {
            bestPctCorrect = PercentageSplit66WeightedPctCorrect;
            bestPctCorrectMethod = "Hold out (" + percent + "%)";
        }



        fileWriter.write("**Results** :");

        fileWriter.write("\n\n");

        fileWriter.write("| | Kapp | PctCorrect | Precision | Recall | F-Measure |\n" +
                "| --- | --- | --- | --- | --- | --- |\n");

        fileWriter.write("| Training set : | " +
                String.format("%.3f", TrainingSetWeightedKappa) + " | " +
                String.format("%.3f", TrainingSetWeightedPctCorrect) + " | " +
                String.format("%.3f", TrainingSetWeightedPrecision) + " | " +
                String.format("%.3f", TrainingSetWeightedRecall) + " | " +
                String.format("%.3f", TrainingSetWeightedFMeasure) + " | \n");

        fileWriter.write("| Cross-validation (10) : | " +
                String.format("%.3f", CrossValidationWeightedKappa) + " | " +
                String.format("%.3f", CrossValidationWeightedPctCorrect) + " | " +
                String.format("%.3f", CrossValidationWeightedPrecision) + " | " +
                String.format("%.3f", CrossValidationWeightedRecall) + " | " +
                String.format("%.3f", CrossValidationWeightedFMeasure) + " | \n");

        fileWriter.write("| Leave one out (" + dataset.numInstances() + ") : | " +
                String.format("%.3f", CrossValidationAllWeightedKappa) + " | " +
                String.format("%.3f", CrossValidationAllWeightedPctCorrect) + " | " +
                String.format("%.3f", CrossValidationAllWeightedPrecision) + " | " +
                String.format("%.3f", CrossValidationAllWeightedRecall) + " | " +
                String.format("%.3f", CrossValidationAllWeightedFMeasure) + " | \n");

        fileWriter.write("| Hold out (" + percent + "%) : | " +
                String.format("%.3f", PercentageSplit66WeightedKappa) + " | " +
                String.format("%.3f", PercentageSplit66WeightedPctCorrect) + " | " +
                String.format("%.3f", PercentageSplit66WeightedPrecision) + " | " +
                String.format("%.3f", PercentageSplit66WeightedRecall) + " | " +
                String.format("%.3f", PercentageSplit66WeightedFMeasure) + " | \n");




        fileWriter.write("**Comparaison** :\n\n" +
                "comparaison à l'aide du **Kappa** :\n\n" +
                "le meilleur résultat est : " + bestKappaMethod + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestKappa) + "\n\n" +
                "comparaison à l'aide du **pct Correct** :\n\n" +
                "le meilleur résultat est : " + bestPctCorrectMethod + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestPctCorrect) + "\n\n" +
                "comparaison à l'aide du **F-Measure** :\n\n" +
                "le meilleur résultat est : " + bestFMeasureMethod + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestPrecision) + "\n\n" +
                "comparaison à l'aide du **Recall** :\n\n" +
                "le meilleur résultat est : " + bestRecallMethod + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestRecall) + "\n\n" +
                "comparaison à l'aide du **Precision** :\n\n" +
                "le meilleur résultat est " + bestPrecisionMethod + " \n" +
                "la valeur qu'il a donné : " + String.format("%.3f", bestFMeasure) + "\n\n");
    }

    public static void dataSetInformation(FileWriter fileWriter, Instances dataset) throws IOException {
        dataset.setClassIndex(dataset.numAttributes() - 1);

        fileWriter.write("### DataSet : " + dataset.relationName() + "\n\n");
        System.out.println("### DataSet : " + dataset.relationName());

        fileWriter.write("\n" +
                "**Le nombre d'attributs** : " + dataset.numAttributes() + "\n\n" +
                "**Le nombre d'instances** : " + dataset.numInstances() +
                "\n\n");

        fileWriter.write("#### Résumé de dataset\n\n" +
                "```Summary \n" +
                dataset.toSummaryString() +
                "```\n\n");
    }
}
