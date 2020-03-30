package tp01;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;

import static tp.solver.TPSolver.doTheTP;

public class Main {
    public static void main(String[] args) throws Exception {

        File tp = new File("docs/TPs/TP01.md");

        if (tp.createNewFile() || tp.exists()) {

            FileWriter fileWriter = new FileWriter(tp);

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


            doTheTP(fileWriter, datasetPaths, classifiers);

            fileWriter.close();
        }


    }

}
