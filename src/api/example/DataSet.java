package api.example;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class DataSet {

    public static void main(String[] args) throws Exception {

        //the path to the dataset arff file
        String datasetPath = "datasets/weather.nominal.arff";

        //A DataSource Object that will give us the dataset
        //make sure to add throws Exception to the method  signature
        DataSource source = new DataSource(datasetPath);

        //An Instances object that will have the dataset
        Instances dataset = source.getDataSet();

        //display the name of the dataset
        System.out.println("### DataSet : " + dataset.relationName());

        //display the number of attributes and the number of instances
        System.out.println("\n" +
                "**Le nombre d'attributs** : " + dataset.numAttributes() + "\n" +
                "**Le nombre d'instances** : " + dataset.numInstances() +
                "\n\n");

        //display a Summary about the dataset
        System.out.println("#### Résumé de dataset\n\n" +
                "```Summary \n" +
                dataset.toSummaryString() +
                "```\n\n");

        //setting the class
        dataset.setClassIndex(dataset.numAttributes() - 1);

        //making the dataset random
        dataset.randomize(new Random(1));

        //create a new evaluation
        Evaluation eval = new Evaluation(dataset);

        //run the evaluation with cross validation
        //using the OneR algothime
        //with 10 folds
        //and the random seed is 1
        eval.crossValidateModel(new OneR(), dataset, 10, new Random(1));

        //display the result
        System.out.println(eval.toSummaryString());

        System.out.println("Precision: " + eval.precision(0) + " | " + eval.precision(1) + " | " + eval.weightedPrecision());
        System.out.println("Recall: " + eval.recall(0) + " | " + eval.recall(1) + " | " + eval.weightedRecall());
        System.out.println("FMeasure: " + eval.fMeasure(0) + " | " + eval.fMeasure(1) + " | " + eval.weightedFMeasure());

    }
}
