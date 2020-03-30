package tp03;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;


import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

import static tp.solver.TPSolver.dataSetInformation;
import static tp.solver.TPSolver.doTheTP;

public class TP03 {
    public static void main(String[] args) throws Exception {
        File tp = new File("docs/TPs/TP03.md");
        if (tp.createNewFile() || tp.exists()) {
            FileWriter fileWriter = new FileWriter(tp);

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

            ConverterUtils.DataSource source = new ConverterUtils.DataSource("datasets/adult.arff");
            Instances dataset = source.getDataSet();

            dataSetInformation(fileWriter, dataset);

            doTheTP(fileWriter, classifiers, dataset);

            fileWriter.write("using the comparison,I found that the best two algorithms are j48 and NaiveBayes, so i will be using them from now on");



            classifiers.clear();
            classifiers.add(j48);
            classifiers.add(naiveBayes);



            fileWriter.write("## Discrétisations supervise et non supervise :\n");
            fileWriter.write("## Discrétisations non supervise :\n");
            fileWriter.write("## intervalle égal :\n");

            for (int i = 2; i < 65; i *= 2) {
                System.out.println("Discrétisations non supervise intervalle égal nombre d'intervalle (" + i + ") ");
                fileWriter.write("## nombre d'intervalle : " + i + "\n");

                Instances unsupDiscretizedDataset;

                Discretize unsupDiscretize = new Discretize();

                unsupDiscretize.setAttributeIndices("first-last");
                unsupDiscretize.setBins(i);
                unsupDiscretize.setInputFormat(dataset);

                unsupDiscretizedDataset = Filter.useFilter(dataset, unsupDiscretize);

                dataSetInformation(fileWriter, unsupDiscretizedDataset);

                doTheTP(fileWriter, classifiers, unsupDiscretizedDataset);
            }

            fileWriter.write("## fréquence égale :\n");

            for (int i = 2; i < 65; i *= 2) {
                System.out.println("Discrétisations non supervise intervalle égal nombre d'intervalle (" + i + ") ");
                fileWriter.write("## nombre d'intervalle : " + i + "\n");

                Instances unsupDiscretizedDataset;

                Discretize unsupDiscretize = new Discretize();

                unsupDiscretize.setAttributeIndices("first-last");
                unsupDiscretize.setBins(i);
                unsupDiscretize.setUseEqualFrequency(true);
                unsupDiscretize.setInputFormat(dataset);

                unsupDiscretizedDataset = Filter.useFilter(dataset, unsupDiscretize);

                dataSetInformation(fileWriter, unsupDiscretizedDataset);

                doTheTP(fileWriter, classifiers, unsupDiscretizedDataset);
            }

            fileWriter.write("## Discrétisations supervise :\n");

            System.out.println("Discrétisations non supervise");

            Instances DiscretizedDataset;

            weka.filters.supervised.attribute.Discretize supDiscretize = new weka.filters.supervised.attribute.Discretize();

            supDiscretize.setAttributeIndices("first-last");
            supDiscretize.setBinRangePrecision(6);
            supDiscretize.setInputFormat(dataset);

            DiscretizedDataset = Filter.useFilter(dataset, supDiscretize);

            dataSetInformation(fileWriter, DiscretizedDataset);

            doTheTP(fileWriter, classifiers, DiscretizedDataset);

            fileWriter.write("\n\n\n\n");


            fileWriter.write("## Sélection d'attributs méthodes filters et wrapper :\n");

            for (Classifier classifier : classifiers) {
                WrapperSubsetEval wrapper = new WrapperSubsetEval();

                wrapper.setClassifier(classifier);
                dataset.setClassIndex(dataset.numAttributes() - 1);
                wrapper.buildEvaluator(dataset);

                BestFirst bestFirst = new BestFirst();

                AttributeSelection attributeSelection = new AttributeSelection();

                attributeSelection.setEvaluator(wrapper);
                attributeSelection.setSearch(bestFirst);
                attributeSelection.setInputFormat(dataset);

                Instances attributeSelectedDataset = Filter.useFilter(dataset, attributeSelection);

                dataSetInformation(fileWriter, attributeSelectedDataset);

                doTheTP(fileWriter, classifier, attributeSelectedDataset);
            }



            fileWriter.write("## prepossessing\n");
            fileWriter.write("## replacing missing values");


            ReplaceMissingValues replaceMissingValues = new ReplaceMissingValues();
            replaceMissingValues.setInputFormat(dataset);

            Instances datasetWithValuesReplaced = Filter.useFilter(dataset, replaceMissingValues);


            dataSetInformation(fileWriter, datasetWithValuesReplaced);

            doTheTP(fileWriter, classifiers, datasetWithValuesReplaced);


            fileWriter.close();

        }
    }
}
