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
            datasetPaths.add("datasets/glass.arff");
            datasetPaths.add("datasets/breast-cancer.arff");
            datasetPaths.add("datasets/contact-lenses.arff");
            datasetPaths.add("datasets/cpu.arff");
            datasetPaths.add("datasets/iris.arff");
            datasetPaths.add("datasets/vote.arff");


            OneR oneR = new OneR();
            NaiveBayes naiveBayes = new NaiveBayes();
            IBk iBk = new IBk();
            J48 j48 = new J48();
            Id3 id3 = new Id3();

            ArrayList<Classifier> classifiers = new ArrayList<>();

            classifiers.add(oneR);
            classifiers.add(naiveBayes);
            classifiers.add(iBk);
            classifiers.add(j48);
            classifiers.add(id3);


            for (String datasetPath : datasetPaths) {

                source = new DataSource(datasetPath);
                Instances dataset = source.getDataSet();
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


                        // leave one out
                        eval = new Evaluation(dataset);
                        eval.crossValidateModel(classifier, dataset, dataset.numInstances(), new Random(1));

                        double CrossValidationAllWeightedPrecision = eval.weightedPrecision();
                        double CrossValidationAllWeightedRecall = eval.weightedRecall();
                        double CrossValidationAllWeightedFMeasure = eval.weightedFMeasure();


                        // hold out
                        int percent = 70;

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


                        // bootstrap

                        source = new DataSource(datasetPath);
                        dataset = source.getDataSet();
                        dataset.setClassIndex(dataset.numAttributes() - 1);

                        Random random = new Random(1);
                        int[] ints = random.ints(dataset.numInstances()).toArray();

                        for (int i = 0; i < ints.length; i++) {
                            ints[i] = Math.abs(ints[i]) % ints.length;
                        }

                        Instances bootstrapTrain = new Instances(dataset,0);

                        for (int i = 0; i < ints.length; i++) {
                            bootstrapTrain.add(dataset.instance(ints[i]));
                        }

                        classifier.buildClassifier(bootstrapTrain);
                        eval = new Evaluation(bootstrapTrain);

                        Instances bootstrapTest = new Instances(dataset,0);

                        for (int i = 0; i < ints.length; i++) {
                            boolean has = false;

                            for (int j = 0; j < ints.length; j++) {
                                if(ints[j] == i) {
                                    has = true;
                                    break;
                                }
                            }

                            if (!has) {
                                bootstrapTrain.add(dataset.instance(ints[i]));
                            }
                        }

                        eval.evaluateModel(classifier, bootstrapTrain);

                        double bootstrapWeightedPrecision = eval.weightedPrecision();
                        double bootstrapWeightedRecall = eval.weightedRecall();
                        double bootstrapWeightedFMeasure = eval.weightedFMeasure();

                        eval.evaluateModel(classifier, bootstrapTest);

                        bootstrapWeightedPrecision = bootstrapWeightedPrecision * 0.63 + eval.weightedPrecision() * 0.37;
                        bootstrapWeightedRecall = bootstrapWeightedRecall * 0.63 + eval.weightedRecall() * 0.37;
                        bootstrapWeightedFMeasure = bootstrapWeightedFMeasure * 0.63 + eval.weightedFMeasure() * 0.37;


                        // base d'apprentissage

                        classifier.buildClassifier(dataset);
                        eval = new Evaluation(dataset);
                        eval.evaluateModel(classifier, dataset);

                        double TrainingSetWeightedPrecision = eval.weightedPrecision();
                        double TrainingSetWeightedRecall = eval.weightedRecall();
                        double TrainingSetWeightedFMeasure = eval.weightedFMeasure();


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

                        /*
                        if (bestPrecision < TrainingSetWeightedPrecision) {
                            bestPrecision = TrainingSetWeightedPrecision;
                            bestPrecisionMethod = "Base d'apprentissage";
                        }
                        if (bestRecall < TrainingSetWeightedRecall) {
                            bestRecall = TrainingSetWeightedRecall;
                            bestRecallMethod = "Base d'apprentissage";
                        }
                        if (bestFMeasure < TrainingSetWeightedFMeasure) {
                            bestFMeasure = TrainingSetWeightedFMeasure;
                            bestFMeasureMethod = "Base d'apprentissage";
                        }
                        */

                        /*
                        if (bestPrecision < bootstrapWeightedPrecision) {
                            bestPrecision = bootstrapWeightedPrecision;
                            bestPrecisionMethod = "Bootstrap";
                        }
                        if (bestRecall < bootstrapWeightedRecall) {
                            bestRecall = bootstrapWeightedRecall;
                            bestRecallMethod = "Bootstrap";
                        }
                        if (bestFMeasure < bootstrapWeightedFMeasure) {
                            bestFMeasure = bootstrapWeightedFMeasure;
                            bestFMeasureMethod = "Bootstrap";
                        }
                        */

                        fileWriter.write("**Results** :");

                        fileWriter.write("\n\n");

                        fileWriter.write("|  | Precision | Recall | F-Measure |\n" +
                                "| --- | --- | --- | --- |\n");

                        fileWriter.write("| Training set : | " +
                                String.format("%.3f", TrainingSetWeightedPrecision) + " | " +
                                String.format("%.3f", TrainingSetWeightedRecall) + " | " +
                                String.format("%.3f", TrainingSetWeightedFMeasure) + " | \n");

                        fileWriter.write("| Cross-validation (10) : | " +
                                String.format("%.3f", CrossValidationWeightedPrecision) + " | " +
                                String.format("%.3f", CrossValidationWeightedRecall) + " | " +
                                String.format("%.3f", CrossValidationWeightedFMeasure) + " | \n");

                        fileWriter.write("| Leave one out (" + dataset.numInstances() + ") : | " +
                                String.format("%.3f", CrossValidationAllWeightedPrecision) + " | " +
                                String.format("%.3f", CrossValidationAllWeightedRecall) + " | " +
                                String.format("%.3f", CrossValidationAllWeightedFMeasure) + " | \n");

                        fileWriter.write("| Hold out (" + percent + "%) : | " +
                                String.format("%.3f", PercentageSplit66WeightedPrecision) + " | " +
                                String.format("%.3f", PercentageSplit66WeightedRecall) + " | " +
                                String.format("%.3f", PercentageSplit66WeightedFMeasure) + " | \n");


                        /*
                        fileWriter.write("| Bootstrap : | " +
                                String.format("%.3f", bootstrapWeightedPrecision) + " | " +
                                String.format("%.3f", bootstrapWeightedRecall) + " | " +
                                String.format("%.3f", bootstrapWeightedFMeasure) + " | \n\n\n");
                         */


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
