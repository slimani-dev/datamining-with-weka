package tp01;

/**
 *
 * @author Chaimaa HM
 */import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.lazy.IBk;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.stemmers.Stemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TP45 {
    public static void main(String[] args) throws Exception{
        DataSource source = new DataSource("C:\\Users\\LE NOVO\\Desktop\\Rapport Weka\\tp4\\dataset_9_autos_replace.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes()-1);
        StringToWordVector filter = new StringToWordVector();

        try {
            //filter.setOutputWordCounts(true);
            //  filter.setIDFTransform(true);
            // filter.setTFTransform(true);
            filter.setLowerCaseTokens(true);
            //WordTokenizer token = new WordTokenizer();
            //filter.setTokenizer(token);
            //  Stemmer s=new LovinsStemmer();
            //  filter.setStemmer(s);
            //Rainbow ss= new Rainbow();
            // filter.setStopwordsHandler(ss);
            filter.setWordsToKeep(1000);

            filter.setInputFormat(dataset);
        } catch (Exception e1) {
            e1.printStackTrace();
        }
        dataset = Filter.useFilter(dataset, filter);

        // System.out.print(dataset.numInstances());
        IBk ib = new IBk();
        //      NaiveBayes nb= new NaiveBayes();
        Bagging mBagg= new Bagging();

        mBagg.setClassifier(ib);

        mBagg.setNumIterations(10);


        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(mBagg, dataset, 10, new Random(1));
        System.out.println(eval.toClassDetailsString("\nDetailed Accuracy By Class\n\n"));
        System.out.println(eval.toMatrixString("\n Confusion Matrix\n\n"));

    }
}