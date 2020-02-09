# API Java

La bibliothèque de Weka fournit une large collection d'algorithmes d'apprentissage automatique, implémentés en Java.

## Exigences:
il y a des exigences pour commencer à utiliser l'API Weka en Java

 - Kit de développement Java (JDK), vous pouvez télécharger à partir de http://www.oracle.com/technetwork/java/javase/downloads/index.html.
 - Un IDE comme [Netbeans](https://netbeans.org), [Eclipse](https://www.eclipse.org) ou [IDEA](https://www.jetbrains.com/idea/download/)
 - [Bibliothèque Weka](https://www.cs.waikato.ac.nz/ml/weka/).
 
Enfin, assurez-vous d'importer le fichier weka.jar dans votre projet

> vérifier les liens pour savoir comment faire ou chercher dans google : [IDEA](https://www.jetbrains.com/help/idea/library.html), [Eclipse](https://help.eclipse.org/2019-12/index.jsp?topic=%2Forg.eclipse.jdt.doc.user%2Freference%2Fpreferences%2Fjava%2Fbuildpath%2Fref-preferences-user-libraries.htm)...

## Ouverture d'un dataset:

Pour ouvrir un dataset, on utilise deux classes une classe `DataSource` et une 
classe `Instances`. La classe `DataSource` ouvrira le fichier dataset arff et
on l'utilis pour créer un objet `Instances`, cet objet Instances contient notre dataset

```java
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

    }
}

```
### Methods de classe `Instances` 
L'étape suivante consiste à afficher des informations sur le dataset comme le nom,
 le nombre de lignes et un résumé

afficher le nom de l'ensemble de données

```java
...

//display the name of the dataset
System.out.println("### DataSet : " + dataset.relationName());

...
```

*resultat*

```text
### DataSet : weather.symbolic
```

Le nombre d'attributs & Le nombre d'instances

```java
...


//display the number of attributes and the number of instances
System.out.println("\n" +
        "**Le nombre d'attributs** : " + dataset.numAttributes() + "\n" +
        "**Le nombre d'instances** : " + dataset.numInstances() +
        "\n\n");

...
```

*resultat*

```text
**Le nombre d'attributs** : 5
**Le nombre d'instances** : 14
```

Résumé de dataset

```java
...

//display a Summary about the dataset
System.out.println("#### Résumé de dataset\n\n" +
        "```Summary \n" +
        dataset.toSummaryString() +
        "```\n\n");

...
```

*resultat*

```text
#### Résumé de dataset

Summary 
Relation Name:  weather.symbolic
Num Instances:  14
Num Attributes: 5

     Name                      Type  Nom  Int Real     Missing      Unique  Dist
1 outlook                    Nom 100%   0%   0%     0 /  0%     0 /  0%     3 
2 temperature                Nom 100%   0%   0%     0 /  0%     0 /  0%     3 
3 humidity                   Nom 100%   0%   0%     0 /  0%     0 /  0%     2 
4 windy                      Nom 100%   0%   0%     0 /  0%     0 /  0%     2 
5 play                       Nom 100%   0%   0%     0 /  0%     0 /  0%     2 
```


Choisir la classe

```java
//setting the class
dataset.setClassIndex(dataset.numAttributes() - 1);
```

Rendre le dataset aléatoire

```java
//making the dataset random with the seed 1
dataset.randomize(new Random(1));
```


## Evaluer le dataset

nous utilisons la classe `Evaluation` pour l'évaluation

```java
//create a new evaluation
Evaluation eval = new Evaluation(dataset);
```

exécuter l'évaluation avec validation croisée en utilisant 
l'algothime OneR avec 10 plis et la graine aléatoire est 1

```java
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
```

*resultat*
```text
Correctly Classified Instances           7               50      %
Incorrectly Classified Instances         7               50      %
Kappa statistic                          0.0392
Mean absolute error                      0.5   
Root mean squared error                  0.7071
Relative absolute error                105      %
Root relative squared error            143.3236 %
Total Number of Instances               14     

Precision: 0.6666666666666666 | 0.375 | 0.5625
Recall: 0.4444444444444444 | 0.6 | 0.5
FMeasure: 0.5333333333333333 | 0.4615384615384615 | 0.5076923076923077
```


> l'exemple de classe dans [ici](https://github.com/Mohamed-SM/datamining-with-weka/blob/master/src/api/example/DataSet.java)