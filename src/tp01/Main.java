package tp01;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {

        File tp = new File("docs/TPs/TP01.md");

        if (tp.createNewFile() || tp.exists()) {

            FileWriter fileWriter = new FileWriter(tp);

            DataSource source;

            ArrayList<String> datasetPaths = new ArrayList<>();
            datasetPaths.add("datasets/weather.nominal.arff");
            datasetPaths.add("datasets/breast-cancer.arff");
            datasetPaths.add("datasets/contact-lenses.arff");
            datasetPaths.add("datasets/cpu.arff");

            ArrayList<Instances> datasets = new ArrayList<>();


            for (String datasetPath : datasetPaths) {
                source = new DataSource(datasetPath);
                Instances dataset = source.getDataSet();
                dataset.setClassIndex(dataset.numAttributes() - 1);
                datasets.add(dataset);
            }


            ArrayList<Classifier> classifiers = new ArrayList<>();

            OneR oneR = new OneR();
            classifiers.add(oneR);

            NaiveBayes naiveBayes = new NaiveBayes();
            classifiers.add(naiveBayes);

            IBk iBk = new IBk();
            classifiers.add(iBk);

            J48 j48 = new J48();
            classifiers.add(j48);

            Id3 id3 = new Id3();
            classifiers.add(id3);


            for (Instances dataset : datasets) {
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

                double bestPrecisionForAll = 0;
                double bestRecallForAll = 0;
                double bestFMeasureForAll = 0;

                String bestPrecisionMethodForAll = "";
                String bestRecallMethodForAll = "";
                String bestFMeasureMethodForAll = "";


                String bestPrecisionAlgorithmForAll = "";
                String bestRecallAlgorithmForAll = "";
                String bestFMeasureAlgorithmForAll = "";


                for (Classifier classifier : classifiers) {
                    try {
                        double bestPrecision = 0;
                        double bestRecall = 0;
                        double bestFMeasure = 0;

                        String bestPrecisionMethod = "";
                        String bestRecallMethod = "";
                        String bestFMeasureMethod = "";

                        Evaluation eval;

                        System.out.println("Algo: " + classifier.getClass().getSimpleName());
                        fileWriter.write("\n\n");
                        fileWriter.write("#### Algo: " + classifier.getClass().getSimpleName());
                        fileWriter.write("\n\n");


                        eval = new Evaluation(dataset);
                        eval.crossValidateModel(classifier, dataset, 10, new Random(1));

                        double CrossValidationWeightedPrecision = eval.weightedPrecision();
                        double CrossValidationWeightedRecall = eval.weightedRecall();
                        double CrossValidationWeightedFMeasure = eval.weightedFMeasure();

                        eval = new Evaluation(dataset);
                        eval.crossValidateModel(classifier, dataset, dataset.numInstances(), new Random(1));

                        double CrossValidationAllWeightedPrecision = eval.weightedPrecision();
                        double CrossValidationAllWeightedRecall = eval.weightedRecall();
                        double CrossValidationAllWeightedFMeasure = eval.weightedFMeasure();

                        dataset.randomize(new Random(1));

                        RemovePercentage rp = new RemovePercentage();
                        rp.setPercentage(66);
                        rp.setInvertSelection(true);
                        rp.setInputFormat(dataset);
                        Instances train = Filter.useFilter(dataset, rp);

                        rp.setPercentage(66);
                        rp.setInvertSelection(false);
                        rp.setInputFormat(dataset);
                        Instances test = Filter.useFilter(dataset, rp);

                        classifier.buildClassifier(train);
                        eval = new Evaluation(train);
                        eval.evaluateModel(classifier, test);

                        double PercentageSplit66WeightedPrecision = eval.weightedPrecision();
                        double PercentageSplit66WeightedRecall = eval.weightedRecall();
                        double PercentageSplit66WeightedFMeasure = eval.weightedFMeasure();


                        classifier.buildClassifier(dataset);
                        eval = new Evaluation(dataset);
                        eval.evaluateModel(classifier, dataset);

                        double TrainingSetWeightedPrecision = eval.weightedPrecision();
                        double TrainingSetWeightedRecall = eval.weightedRecall();
                        double TrainingSetWeightedFMeasure = eval.weightedFMeasure();


                        if (bestPrecision < CrossValidationWeightedPrecision) {
                            bestPrecision = CrossValidationWeightedPrecision;
                            bestPrecisionMethod = "Cross-validation (10)";
                        }
                        if (bestRecall < CrossValidationWeightedRecall) {
                            bestRecall = CrossValidationWeightedRecall;
                            bestRecallMethod = "Cross-validation (10)";
                        }
                        if (bestFMeasure < CrossValidationWeightedFMeasure) {
                            bestFMeasure = CrossValidationWeightedFMeasure;
                            bestFMeasureMethod = "Cross-validation (10)";
                        }


                        if (bestPrecision < CrossValidationAllWeightedPrecision) {
                            bestPrecision = CrossValidationAllWeightedPrecision;
                            bestPrecisionMethod = "Cross-validation (" + dataset.numInstances() + ")";
                        }
                        if (bestRecall < CrossValidationAllWeightedRecall) {
                            bestRecall = CrossValidationAllWeightedRecall;
                            bestRecallMethod = "Cross-validation (" + dataset.numInstances() + ")";
                        }
                        if (bestFMeasure < CrossValidationAllWeightedFMeasure) {
                            bestFMeasure = CrossValidationAllWeightedFMeasure;
                            bestFMeasureMethod = "Cross-validation (" + dataset.numInstances() + ")";
                        }


                        if (bestPrecision < TrainingSetWeightedPrecision) {
                            bestPrecision = TrainingSetWeightedPrecision;
                            bestPrecisionMethod = "Training set";
                        }
                        if (bestRecall < TrainingSetWeightedRecall) {
                            bestRecall = TrainingSetWeightedRecall;
                            bestRecallMethod = "Training set";
                        }
                        if (bestFMeasure < TrainingSetWeightedFMeasure) {
                            bestFMeasure = TrainingSetWeightedFMeasure;
                            bestFMeasureMethod = "Training set";
                        }


                        if (bestPrecision < PercentageSplit66WeightedPrecision) {
                            bestPrecision = PercentageSplit66WeightedPrecision;
                            bestPrecisionMethod = "Percentage Split (66%)";
                        }
                        if (bestRecall < PercentageSplit66WeightedRecall) {
                            bestRecall = PercentageSplit66WeightedRecall;
                            bestRecallMethod = "Percentage Split (66)";
                        }
                        if (bestFMeasure < eval.weightedFMeasure()) {
                            bestFMeasure = PercentageSplit66WeightedFMeasure;
                            bestFMeasureMethod = "Percentage Split (66)";
                        }

                        fileWriter.write("**Results** :");

                        fileWriter.write("\n\n");

                        fileWriter.write("|  | Precision | Recall | F-Measure |\n" +
                                "| --- | --- | --- | --- |\n");

                        fileWriter.write("| Cross-validation (10) : | " +
                                String.format("%.3f", CrossValidationWeightedPrecision) + " | " +
                                String.format("%.3f", CrossValidationWeightedRecall) + " | " +
                                String.format("%.3f", CrossValidationWeightedFMeasure) + " | \n");

                        fileWriter.write("| Cross-validation (" + dataset.numInstances() + ") : | " +
                                String.format("%.3f", CrossValidationAllWeightedPrecision) + " | " +
                                String.format("%.3f", CrossValidationAllWeightedRecall) + " | " +
                                String.format("%.3f", CrossValidationAllWeightedFMeasure) + " | \n");


                        fileWriter.write("| Training set : | " +
                                String.format("%.3f", TrainingSetWeightedPrecision) + " | " +
                                String.format("%.3f", TrainingSetWeightedRecall) + " | " +
                                String.format("%.3f", TrainingSetWeightedFMeasure) + " | \n");

                        fileWriter.write("| Percentage Split (66%) : | " +
                                String.format("%.3f", PercentageSplit66WeightedPrecision) + " | " +
                                String.format("%.3f", PercentageSplit66WeightedRecall) + " | " +
                                String.format("%.3f", PercentageSplit66WeightedFMeasure) + " | \n\n\n");


                        fileWriter.write("**Comparaison** :\n\n" +
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

                    } catch (Exception e) {
                        fileWriter.write("Une **erreur** s'est produite lors de l'évaluation\n\n");
                        fileWriter.write("!>" + e.getMessage() + "\n\n");
                    }

                }

                fileWriter.write("#### Comparaison entre algorithmes :\n\n" +
                        "comparaison à l'aide du **F-Measure** :\n\n" +
                        "le meilleur algorithme est : **" + bestFMeasureAlgorithmForAll + "** \n\n" +
                        "le meilleur résultat est : " + bestFMeasureMethodForAll + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestPrecisionForAll) + "\n\n\n" +
                        "comparaison à l'aide du **Recall** :\n\n" +
                        "le meilleur algorithme est : **" + bestRecallAlgorithmForAll + "** \n\n" +
                        "le meilleur résultat est : " + bestRecallMethodForAll + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestRecallForAll) + "\n\n\n" +
                        "comparaison à l'aide du **Precision** :\n\n" +
                        "le meilleur algorithme est : **" + bestPrecisionAlgorithmForAll + "** \n\n" +
                        "le meilleur résultat est " + bestPrecisionMethodForAll + " \n" +
                        "la valeur qu'il a donné : " + String.format("%.3f", bestFMeasureForAll) + "\n\n\n");


            }

            fileWriter.close();
        }


    }
}
